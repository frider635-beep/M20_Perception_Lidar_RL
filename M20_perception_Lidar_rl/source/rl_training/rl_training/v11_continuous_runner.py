"""RSL-RL 5.x compatibility runner and ABI gate for M20 V11."""

from __future__ import annotations

import os

import torch

from rsl_rl.runners import OnPolicyRunner

class V11ContinuousRunner(OnPolicyRunner):
    """Bridge Isaac Lab's legacy PPO config to the installed RSL-RL 5.x ABI.

    V11 intentionally uses the native shared MLP for both front legs.  The
    bridge is needed only because this repository's Isaac Lab config schema
    still emits the pre-4.x ``policy`` block whereas the installed PPO expects
    separate ``actor`` and ``critic`` model blocks.
    """

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        cfg = dict(train_cfg)
        require_pretrained = bool(cfg.pop("require_pretrained_checkpoint", False))
        # Isaac Lab 0.54's forward-compatible dataclass always serializes
        # ``beta``; the installed RSL-RL 5.x PPO constructor has no such arg.
        cfg["algorithm"] = dict(cfg["algorithm"])
        cfg["algorithm"].pop("beta", None)
        policy = dict(cfg.pop("policy"))
        policy_class = policy.pop("class_name", None)
        if policy_class not in ("V11ContinuousActorCritic", "V135RecurrentActorCritic", "V135ResidualGRUActor"):
            raise RuntimeError("V11 runner received the wrong policy configuration")
        recurrent_actor = policy_class == "V135RecurrentActorCritic"
        residual_gru_actor = policy_class == "V135ResidualGRUActor"
        init_std = policy.pop("init_noise_std")
        std_type = policy.pop("noise_std_type")
        state_dependent_std = policy.pop("state_dependent_std", False)
        if state_dependent_std:
            raise RuntimeError("V11 requires a state-independent log action standard deviation")
        actor_normalization = policy.pop("actor_obs_normalization")
        critic_normalization = policy.pop("critic_obs_normalization")
        actor_hidden = policy.pop("actor_hidden_dims")
        critic_hidden = policy.pop("critic_hidden_dims")
        activation = policy.pop("activation")
        rnn_type = policy.pop("rnn_type", None)
        rnn_hidden_dim = policy.pop("rnn_hidden_dim", None)
        rnn_num_layers = policy.pop("rnn_num_layers", None)
        if recurrent_actor:
            if rnn_type != "gru" or int(rnn_hidden_dim or 0) <= 0 or int(rnn_num_layers or 0) <= 0:
                raise RuntimeError("V13.5 requires a configured GRU actor")
        elif residual_gru_actor:
            if rnn_type != "gru" or int(rnn_hidden_dim or 0) <= 0 or int(rnn_num_layers or 0) <= 0:
                raise RuntimeError("V13.5-Final residual GRU actor requires configured GRU fields")
        elif any(value is not None for value in (rnn_type, rnn_hidden_dim, rnn_num_layers)):
            raise RuntimeError("Only the V13.5 recurrent actor may configure RNN fields")
        # Isaac Lab 0.54 serializes this legacy field even though V11 has no
        # depth-image encoder.  It must stay disabled rather than silently
        # changing the actor ABI.
        include_depth_input = policy.pop("include_depth_input", False)
        if include_depth_input:
            raise RuntimeError("V11 does not permit a hidden depth-image actor input")
        if policy:
            raise RuntimeError(f"V11 runner received unsupported policy fields: {sorted(policy)}")
        cfg["obs_groups"] = {"actor": ["policy"], "critic": ["critic"]}

        # Actor configuration
        if residual_gru_actor:
            cfg["actor"] = {
                "class_name": "rl_training.v135_final_actor:V135ResidualGRUModel",
                "hidden_dims": actor_hidden,
                "activation": activation,
                "obs_normalization": actor_normalization,
                "distribution_cfg": {
                    "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                    "init_std": init_std,
                    "std_type": std_type,
                },
                "gru_hidden_dim": int(rnn_hidden_dim),
                "gru_input_dim": 32,
                "residual_scale": 0.20,
            }
        else:
            cfg["actor"] = {
                "class_name": (
                    "rl_training.v135_recurrent_model:V135RNNModel"
                    if recurrent_actor else "rsl_rl.models:MLPModel"
                ),
                "hidden_dims": actor_hidden,
            "activation": activation,
            "obs_normalization": actor_normalization,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                "init_std": init_std,
                "std_type": std_type,
            },
        }
        if recurrent_actor:
            cfg["actor"].update({
                "rnn_type": rnn_type,
                "rnn_hidden_dim": int(rnn_hidden_dim),
                "rnn_num_layers": int(rnn_num_layers),
            })
        cfg["critic"] = {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": critic_hidden,
            "activation": activation,
            "obs_normalization": critic_normalization,
        }
        super().__init__(env, cfg, log_dir=log_dir, device=device)
        if require_pretrained:
            checkpoint = os.environ.get("V11_SOFTALT_CHECKPOINT", "").strip()
            if not checkpoint:
                raise RuntimeError(
                    "V11 SoftAlt refuses random initialization; set V11_SOFTALT_CHECKPOINT "
                    "to the selected ABI-compatible V11 checkpoint."
                )
            checkpoint = os.path.realpath(checkpoint)
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(f"V11 SoftAlt checkpoint does not exist: {checkpoint}")
            optimizer_was_fresh = not self.alg.optimizer.state
            self.load(
                checkpoint,
                load_cfg={
                    "actor": True,
                    "critic": True,
                    "optimizer": False,
                    "iteration": False,
                    "rnd": False,
                },
            )
            loaded = torch.load(checkpoint, weights_only=False, map_location="cpu")
            for model_name, model in (("actor", self.alg.actor), ("critic", self.alg.critic)):
                expected = loaded[f"{model_name}_state_dict"]
                actual = model.state_dict()
                if actual.keys() != expected.keys() or any(
                    not torch.equal(actual[key].detach().cpu(), expected[key].detach().cpu())
                    for key in actual
                ):
                    raise RuntimeError(f"V11 SoftAlt {model_name} did not exactly load from checkpoint.")
            if not optimizer_was_fresh or self.alg.optimizer.state:
                raise RuntimeError("V11 SoftAlt optimizer must remain newly initialized after checkpoint load.")
            print(f"V11_SOFTALT_LOADED_CHECKPOINT={checkpoint}", flush=True)
            print("V11_SOFTALT_ACTOR_FROM_CHECKPOINT=YES", flush=True)
            print("V11_SOFTALT_CRITIC_FROM_CHECKPOINT=YES", flush=True)
            print("V11_SOFTALT_OPTIMIZER_FRESH=YES", flush=True)
            configured_lr = float(cfg["algorithm"]["learning_rate"])
            optimizer_lrs = {float(group["lr"]) for group in self.alg.optimizer.param_groups}
            if optimizer_lrs != {configured_lr}:
                raise RuntimeError(
                    f"V11 SoftAlt optimizer LR mismatch: expected {configured_lr}, got {sorted(optimizer_lrs)}"
                )
            if self.current_learning_iteration != 0:
                raise RuntimeError(
                    "V11 SoftAlt must start a new objective at iteration 0, got "
                    f"{self.current_learning_iteration}."
                )
            print(f"V11_SOFTALT_ACTUAL_LR={configured_lr:.4f}", flush=True)
            print(f"V11_SOFTALT_OPTIMIZER_LR={next(iter(optimizer_lrs)):.4f}", flush=True)
            print(f"V11_SOFTALT_START_ITERATION={self.current_learning_iteration}", flush=True)
        actor = self.alg.actor
        action_dim = actor.distribution.output_dim if actor.distribution is not None else -1
        if (actor.obs_dim, action_dim) != (63, 16):
            raise RuntimeError("V11 runner dimension gate failed")
        expected_critic_dim = 574 if (recurrent_actor or residual_gru_actor) else self.alg.critic.obs_dim
        if self.alg.critic.obs_dim != expected_critic_dim:
            raise RuntimeError(
                f"V13.5 critic dimension gate failed: expected {expected_critic_dim}, "
                f"got {self.alg.critic.obs_dim}"
            )
        self._write_actor_manifest(log_dir)
        print("V11_ACTOR_OBS_DIM=63")
        print("V11_ACTION_DIM=16")
        if recurrent_actor or residual_gru_actor:
            print(f"V135_ACTOR_GRU_HIDDEN_DIM={int(rnn_hidden_dim)}")
            print(f"V135_ACTOR_GRU_NUM_LAYERS={int(rnn_num_layers)}")

    @staticmethod
    def _write_actor_manifest(log_dir: str | None) -> None:
        if log_dir is None:
            return
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "actor_observation_manifest.txt"), "w", encoding="ascii") as stream:
            stream.write(
                "M20 V11 actor observation (63D)\\n"
                "0:3 base_ang_vel; 3:6 projected_gravity; 6:9 command; "
                "9:25 joint_pos; 25:41 joint_vel; 41:57 previous_action; "
                "57 B1; 58 H1; 59 D1; 60 B2; 61 H2; 62 D2\\n"
                "Actor has no FL/FR-specific action path.\\n"
            )


__all__ = ["V11ContinuousRunner"]
