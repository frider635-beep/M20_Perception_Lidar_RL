"""Load the saved 63D M20 MLP and its actual observation/action contract."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
import yaml

from saved_config import ConfigLoader, from_checkpoint, resolve
from checkpoint_config import resolve_configs


JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("fl", "fr", "hl", "hr")
               for joint in ("hipx", "hipy", "knee")]
JOINT_NAMES += [f"{leg}_wheel_joint" for leg in ("fl", "fr", "hl", "hr")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SavedPolicy:
    def __init__(self, checkpoint, config_dir=None):
        self.path = Path(checkpoint).resolve()
        params, config_resolution, config_source = resolve_configs(self.path, config_dir)
        self.config_path = params / "env.yaml"
        self.cfg = yaml.load(self.config_path.read_text(), Loader=ConfigLoader)
        agent_path = params / "agent.yaml"
        self.agent = yaml.load(agent_path.read_text(), Loader=ConfigLoader)
        options = self.agent["policy"]
        if options["activation"] != "elu" or options["actor_obs_normalization"]:
            raise ValueError("This runner requires the saved ELU MLP with no observation normalization")
        semantics = from_checkpoint(self.path, {"joint_names": JOINT_NAMES}, config_path=self.config_path)
        self.nominal = np.asarray(semantics["default_joint_pos"], dtype=np.float64)
        self.action_scale = np.asarray(semantics["action_scale"], dtype=np.float64)
        self.period = float(self.cfg["sim"]["dt"]) * int(self.cfg["decimation"])
        self.clip_actions = float(self.agent["clip_actions"])
        self.perception = self.cfg["v11_perception"]
        if self.perception["feature_contract"] != "v11_signed_edges_v2":
            raise ValueError("Expected v11_signed_edges_v2; refusing a guessed perception ABI")
        self.obs_cfg = self.cfg["observations"]["policy"]
        expected = {"base_ang_vel": "base_ang_vel", "projected_gravity": "projected_gravity",
                    "velocity_commands": "generated_commands", "joint_pos": "joint_pos_rel_without_wheel",
                    "joint_vel": "joint_vel_rel", "actions": "last_action", "height_scan": "v11_actor_perception"}
        active = [k for k, v in self.obs_cfg.items() if isinstance(v, dict) and "func" in v]
        if active != list(expected):
            raise ValueError(f"Unexpected observation order: {active}")
        for key, function in expected.items():
            term = self.obs_cfg[key]
            if term["func"].split(":")[-1] != function or term["history_length"] != 0 or term["modifiers"]:
                raise ValueError(f"Unsupported observation term: {key}")
        if self.obs_cfg["base_lin_vel"] is not None:
            raise ValueError("Actual base linear velocity must not enter the 63D actor")
        saved = torch.load(self.path, map_location="cpu", weights_only=True)
        state = saved["actor_state_dict"]
        widths = [63, *options["actor_hidden_dims"], 16]
        layers = []
        for i, (nin, nout) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(nin, nout))
            if i < len(widths) - 2:
                layers.append(nn.ELU())
        self.actor = nn.Sequential(*layers)
        expected_keys = {"mlp." + k for k in self.actor.state_dict()} | {"distribution.log_std_param"}
        if set(state) != expected_keys or any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Unexpected/nonfinite actor state; recurrent/MoE/normalized models are unsupported")
        self.actor.load_state_dict({k[4:]: v for k, v in state.items() if k.startswith("mlp.")}, strict=True)
        self.actor.eval()
        self.iteration = int(saved["iter"])
        self.kp, self.kd, self.effort, self.armature = [], [], [], []
        for name in JOINT_NAMES:
            matching = [v for v in self.cfg["scene"]["robot"]["actuators"].values()
                        if any(re.fullmatch(pattern, name) for pattern in v["joint_names_expr"])]
            if len(matching) != 1:
                raise ValueError(f"Ambiguous/missing actuator configuration for {name}")
            group = matching[0]
            for dest, field in [(self.kp, "stiffness"), (self.kd, "damping"),
                                (self.effort, "effort_limit"), (self.armature, "armature")]:
                dest.extend(resolve(group[field], [name], field))
        self.kp, self.kd, self.effort, self.armature = map(np.asarray, (self.kp, self.kd, self.effort, self.armature))
        if np.any(self.kp[12:] != 0) or np.any(self.effort <= 0):
            raise ValueError("Expected wheel velocity control and positive effort limits")
        self.metadata = dict(checkpoint=str(self.path), checkpoint_sha256=sha256(self.path),
                             config_path=str(self.config_path), agent_path=str(agent_path),
                             config_resolution=config_resolution,
                             config_source_checkpoint=str(config_source) if config_source else None,
                             config_sha256=sha256(self.config_path), agent_sha256=sha256(agent_path),
                             iteration=self.iteration, actor_dimensions=widths,
                             feature_contract=self.perception["feature_contract"],
                             joint_names=JOINT_NAMES, default_joint_pos=self.nominal.tolist(),
                             action_scale=self.action_scale.tolist(), policy_period=self.period,
                             kp=self.kp.tolist(), kd=self.kd.tolist(), effort_limit=self.effort.tolist(),
                             armature=self.armature.tolist(), critic_used=False)

    def observation(self, omega, gravity, command, qpos, qvel, previous, feature):
        relative = qpos - self.nominal
        relative[12:] = 0.
        values = (omega, gravity, command, relative, qvel, previous, feature)
        names = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos",
                 "joint_vel", "actions", "height_scan")
        result = np.concatenate([np.clip(v, *self.obs_cfg[n]["clip"]) * self.obs_cfg[n]["scale"]
                                 for n, v in zip(names, values)]).astype(np.float32)
        if result.shape != (63,) or not np.isfinite(result).all():
            raise RuntimeError("Invalid 63D policy observation")
        return result

    @torch.inference_mode()
    def __call__(self, observation):
        result = self.actor(torch.from_numpy(observation[None])).numpy()[0]
        if result.shape != (16,) or not np.isfinite(result).all():
            raise RuntimeError("Invalid 16D policy action")
        return np.clip(result, -self.clip_actions, self.clip_actions)
