# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Fast ONNX export for Deep Robotics policy — no Isaac Sim required.

Reconstructs the actor MLP directly from a checkpoint and exports it to ONNX,
optionally embedding robot metadata as ONNX model properties.

Examples:
    # Lite3
    python scripts/tools/export_onnx_fast.py \\
        --checkpoint_path logs/rsl_rl/deeprobotics_lite3_rough/2025-01-01_12-00-00/model_5000.pt \\
        --robot lite3 \\
        --output_path exported/lite3_policy.onnx

    # M20
    python scripts/tools/export_onnx_fast.py \\
        --checkpoint_path logs/rsl_rl/deeprobotics_m20_rough/2025-01-01_12-00-00/model_5000.pt \\
        --robot m20 \\
        --output_path exported/m20_policy.onnx
"""

import argparse
import os
import re

import onnx
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Robot constants (joint order matches USD traversal / env config)
# ---------------------------------------------------------------------------

# fmt: off
_LITE3_JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]

_LITE3_LINK_NAMES = [
    "TORSO",
    "FL_HIP", "FR_HIP", "HL_HIP", "HR_HIP",
    "FL_THIGH", "FR_THIGH", "HL_THIGH", "HR_THIGH",
    "FL_SHANK", "FR_SHANK", "HL_SHANK", "HR_SHANK",
    "FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT",
]

# HipX: stiffness=30, damping=1 | HipY: stiffness=30, damping=1 | Knee: stiffness=30, damping=1
_LITE3_STIFFNESS  = [30.0, 30.0, 30.0] * 4
_LITE3_DAMPING    = [1.0,  1.0,  1.0 ] * 4
# Default init pose from DEEPROBOTICS_LITE3_CFG
_LITE3_DEFAULT_POS = [0.0, -0.8, 1.6] * 4
# Action scale: HipX=0.125, HipY=0.25, Knee=0.25
_LITE3_ACTION_SCALE = [0.125, 0.25, 0.25] * 4

LITE3_CFG = {
    "joint_names":       _LITE3_JOINT_NAMES,
    "link_names":        _LITE3_LINK_NAMES,
    "base_link":         "TORSO",
    "stiffness":         _LITE3_STIFFNESS,
    "damping":           _LITE3_DAMPING,
    "default_joint_pos": _LITE3_DEFAULT_POS,
    "action_scale":      _LITE3_ACTION_SCALE,
}

# M20: 12 leg joints + 4 wheel joints = 16 total
_M20_LEG_JOINT_NAMES = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
]
_M20_WHEEL_JOINT_NAMES = [
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
]
_M20_JOINT_NAMES = _M20_LEG_JOINT_NAMES + _M20_WHEEL_JOINT_NAMES

# Leg: stiffness=80, damping=2 | Wheel: stiffness=0, damping=0.6
_M20_STIFFNESS  = [80.0] * 12 + [0.0] * 4
_M20_DAMPING    = [2.0 ] * 12 + [0.6] * 4
# Default init pose from DEEPROBOTICS_M20_CFG
_M20_DEFAULT_POS = [
    0.0, -0.6,  1.0,   # fl: hipx, hipy, knee
    0.0, -0.6,  1.0,   # fr: hipx, hipy, knee
    0.0,  0.6, -1.0,   # hl: hipx, hipy, knee
    0.0,  0.6, -1.0,   # hr: hipx, hipy, knee
    0.0,  0.0,  0.0,  0.0,  # wheels
]
# Action scale: leg hipx=0.125, leg other=0.25, wheel velocity scale=5.0
_M20_ACTION_SCALE = [0.125, 0.25, 0.25] * 4 + [5.0] * 4

M20_CFG = {
    "joint_names":       _M20_JOINT_NAMES,
    "link_names":        None,   # not specified in env config
    "base_link":         "base_link",
    "stiffness":         _M20_STIFFNESS,
    "damping":           _M20_DAMPING,
    "default_joint_pos": _M20_DEFAULT_POS,
    "action_scale":      _M20_ACTION_SCALE,
}
# fmt: on

ROBOT_CONFIGS = {
    "lite3": LITE3_CFG,
    "m20":   M20_CFG,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_actor(state_dict: dict) -> nn.Sequential:
    """Reconstruct actor MLP from checkpoint weights.

    RSL-RL stores actor layers at even indices (0, 2, 4, ...) with ELU
    activations between hidden layers.
    """
    layers: list[nn.Module] = []
    i = 0
    while f"actor.{i}.weight" in state_dict:
        w = state_dict[f"actor.{i}.weight"]
        b = state_dict[f"actor.{i}.bias"]
        linear = nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data.copy_(w)
        linear.bias.data.copy_(b)
        layers.append(linear)
        if f"actor.{i + 2}.weight" in state_dict:  # not the last layer
            layers.append(nn.ELU())
        i += 2
    return nn.Sequential(*layers)


def _build_sequential(state_dict: dict, prefix: str) -> nn.Sequential:
    """Rebuild a Sequential MLP stored below ``prefix`` from its Linear weights."""
    indices = sorted(
        int(match.group(1))
        for key in state_dict
        if (match := re.fullmatch(rf"{re.escape(prefix)}\.(\d+)\.weight", key))
    )
    if not indices:
        raise ValueError(f"No Linear weights found below {prefix!r}.")

    layers: list[nn.Module] = []
    for position, index in enumerate(indices):
        weight = state_dict[f"{prefix}.{index}.weight"]
        bias = state_dict[f"{prefix}.{index}.bias"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if position + 1 != len(indices):
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class _SoftMoEActor(nn.Module):
    """Inference-only copy of V12/V9's soft mixture-of-experts actor."""

    def __init__(self, gate: nn.Sequential, experts: list[nn.Sequential]) -> None:
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList(experts)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        gate_scores = torch.softmax(self.gate(obs), dim=-1)
        expert_outputs = torch.stack([expert(obs) for expert in self.experts], dim=1)
        # Equivalent to the training model's einsum("be,beo->bo", ...), while
        # remaining friendly to the deployment ONNX opset.
        return (gate_scores.unsqueeze(-1) * expert_outputs).sum(dim=1)


def _build_moe_actor(state_dict: dict) -> _SoftMoEActor:
    """Reconstruct the V9/V12 actor from ``mlp.gate`` and ``mlp.experts`` keys."""
    gate = _build_sequential(state_dict, "mlp.gate")
    expert_ids = sorted(
        {
            int(match.group(1))
            for key in state_dict
            if (match := re.fullmatch(r"mlp\.experts\.(\d+)\.\d+\.weight", key))
        }
    )
    if not expert_ids:
        raise ValueError("MoE checkpoint has no expert weights.")
    if expert_ids != list(range(len(expert_ids))):
        raise ValueError(f"MoE expert indices must be contiguous from zero, got {expert_ids}.")

    experts = [_build_sequential(state_dict, f"mlp.experts.{expert_id}") for expert_id in expert_ids]
    output_dims = {expert[-1].out_features for expert in experts if isinstance(expert[-1], nn.Linear)}
    if len(output_dims) != 1:
        raise ValueError("MoE experts do not have a common action dimension.")
    if gate[-1].out_features != len(experts):
        raise ValueError("MoE gate output dimension does not match the number of experts.")
    return _SoftMoEActor(gate, experts)


def _csv(values, decimals: int = 4) -> str:
    fmt = f"{{:.{decimals}f}}"
    return ",".join(fmt.format(v) if isinstance(v, (int, float)) else str(v) for v in values)


def _attach_metadata(onnx_path: str, robot_cfg: dict, checkpoint_path: str) -> None:
    metadata: dict[str, str] = {
        "checkpoint_path":   checkpoint_path,
        "base_link_name":    robot_cfg["base_link"],
        "joint_names":       _csv(robot_cfg["joint_names"]),
        "joint_stiffness":   _csv(robot_cfg["stiffness"]),
        "joint_damping":     _csv(robot_cfg["damping"]),
        "default_joint_pos": _csv(robot_cfg["default_joint_pos"]),
        "action_scale":      _csv(robot_cfg["action_scale"]),
    }
    metadata.update(robot_cfg.get("contract_metadata", {}))
    if robot_cfg["link_names"] is not None:
        metadata["link_names"] = _csv(robot_cfg["link_names"])

    model = onnx.load(onnx_path)
    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = v
        model.metadata_props.append(entry)
    onnx.save(model, onnx_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Deep Robotics RSL-RL policy to ONNX without Isaac Sim."
    )
    parser.add_argument("--checkpoint_path", required=True, help="Path to .pt checkpoint file.")
    parser.add_argument(
        "--robot",
        required=True,
        choices=list(ROBOT_CONFIGS.keys()),
        help="Robot type: 'lite3' or 'm20'.",
    )
    parser.add_argument("--output_path", required=True, help="Output .onnx file path.")
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        default=False,
        help="Skip attaching robot metadata to the ONNX file.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    # --- load checkpoint ---
    print(f"[INFO] Loading: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)

    # RSL-RL 5 stores the actor separately with ``mlp.`` keys.  Older
    # checkpoints store it within ``model_state_dict`` under ``actor.``.
    if "actor_state_dict" in ckpt:
        sd = ckpt["actor_state_dict"]
    else:
        sd = ckpt["model_state_dict"]

    # V9/V12 teachers have a soft MoE under ``mlp.gate``/``mlp.experts``.
    # Keep the established flat MLP path for legacy checkpoints.
    if "mlp.gate.0.weight" in sd:
        actor = _build_moe_actor(sd)
        obs_dim = sd["mlp.gate.0.weight"].shape[1]
        action_dim = actor.experts[0][-1].out_features
        architecture = f"soft_moe_{len(actor.experts)}_experts"
    else:
        # Map RSL-RL 5's flat ``mlp.X`` names to the original format used by
        # the legacy reconstruction helper.
        flat_sd = {
            ("actor." + key[len("mlp.") :] if key.startswith("mlp.") else key): value
            for key, value in sd.items()
        }
        if "actor.0.weight" not in flat_sd:
            raise KeyError(
                "Unsupported actor checkpoint: expected either flat actor.0.weight "
                "or V9/V12 MoE mlp.gate.0.weight."
            )
        actor = _build_actor(flat_sd)
        obs_dim = flat_sd["actor.0.weight"].shape[1]
        action_dim = actor[-1].out_features
        architecture = "flat_mlp"
    print(f"[INFO] architecture={architecture}  obs_dim={obs_dim}  action_dim={action_dim}")

    robot_cfg = ROBOT_CONFIGS[args.robot]
    if not args.no_metadata and args.robot == "m20":
        from m20_export_contract import from_checkpoint
        robot_cfg = from_checkpoint(args.checkpoint_path, robot_cfg)

    # --- build & export actor ---
    actor.eval()

    # --- export to ONNX ---
    dummy_obs = torch.zeros(1, obs_dim)
    torch.onnx.export(
        actor,
        dummy_obs,
        args.output_path,
        export_params=True,
        opset_version=11,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    print(f"[INFO] ONNX saved: {args.output_path}")

    # --- attach metadata ---
    if not args.no_metadata:
        _attach_metadata(args.output_path, robot_cfg, args.checkpoint_path)
        print("[INFO] Robot metadata attached.")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
