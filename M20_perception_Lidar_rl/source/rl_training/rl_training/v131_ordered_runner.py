"""V13.1 retention-test runner with actor-only checkpoint initialization."""

from __future__ import annotations

import os

import torch

from .v11_continuous_runner import V11ContinuousRunner


class V131OrderedRetentionRunner(V11ContinuousRunner):
    """Load exactly one ABI-compatible actor while keeping critic/optimizer fresh."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        cfg = dict(train_cfg)
        require_actor = bool(cfg.pop("require_v131_retention_actor", False))
        super().__init__(env, cfg, log_dir=log_dir, device=device)
        if not require_actor:
            raise RuntimeError("V13.1 retention runner requires an explicit actor checkpoint.")

        checkpoint = os.path.realpath(
            os.environ.get("V131_RETENTION_ACTOR_CHECKPOINT", "").strip()
        )
        if not checkpoint:
            raise RuntimeError(
                "V13.1 retention test requires V131_RETENTION_ACTOR_CHECKPOINT."
            )
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"V13.1 retention actor does not exist: {checkpoint}")

        critic_before = {
            key: value.detach().cpu().clone()
            for key, value in self.alg.critic.state_dict().items()
        }
        if self.alg.optimizer.state:
            raise RuntimeError("V13.1 retention optimizer was not fresh before actor load.")
        self.load(
            checkpoint,
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
        )
        loaded = torch.load(checkpoint, weights_only=False, map_location="cpu")
        expected_actor = loaded["actor_state_dict"]
        actual_actor = self.alg.actor.state_dict()
        if actual_actor.keys() != expected_actor.keys() or any(
            not torch.equal(actual_actor[key].detach().cpu(), expected_actor[key])
            for key in actual_actor
        ):
            raise RuntimeError("V13.1 retention actor did not load exactly.")
        critic_after = self.alg.critic.state_dict()
        if critic_after.keys() != critic_before.keys() or any(
            not torch.equal(critic_after[key].detach().cpu(), critic_before[key])
            for key in critic_after
        ):
            raise RuntimeError("V13.1 retention test unexpectedly loaded the old critic.")
        if self.alg.optimizer.state:
            raise RuntimeError("V13.1 retention optimizer inherited checkpoint state.")
        if self.current_learning_iteration != 0:
            raise RuntimeError("V13.1 retention objective must begin at iteration zero.")

        print(f"V131_RETENTION_ACTOR_CHECKPOINT={checkpoint}", flush=True)
        print("V131_RETENTION_ACTOR_EXACT_LOAD=YES", flush=True)
        print("V131_RETENTION_CRITIC_FRESH=YES", flush=True)
        print("V131_RETENTION_OPTIMIZER_FRESH=YES", flush=True)
        print("V131_RETENTION_START_ITERATION=0", flush=True)


__all__ = ["V131OrderedRetentionRunner"]
