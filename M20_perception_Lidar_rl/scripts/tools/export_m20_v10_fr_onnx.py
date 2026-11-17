#!/usr/bin/env python3
"""Export the deterministic 60D V10 FR actor from an RSL-RL checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


class V10FRActor(nn.Module):
    """Deployment copy of rl_training.v10_unified_policy.UnifiedActor."""

    def __init__(self):
        super().__init__()
        self.shared_trunk = nn.Sequential(
            nn.Linear(58, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
        )
        self.support_head = nn.Linear(128, 13)
        self.fr_head = nn.Sequential(
            nn.Linear(130, 128),
            nn.ELU(),
            nn.Linear(128, 3),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        proprio = observation[..., :57]
        b = observation[..., 57:58]
        height_distance = observation[..., 58:60]
        latent = self.shared_trunk(torch.cat((proprio, b), dim=-1))
        support = self.support_head(latent)
        fr = self.fr_head(torch.cat((latent, height_distance), dim=-1))
        return torch.cat((support[..., :3], fr, support[..., 3:13]), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = V10FRActor().eval()
    actor_state = {
        key.removeprefix("actor."): value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("actor.")
    }
    actor.load_state_dict(actor_state, strict=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.randn(8, 60)
    torch.onnx.export(
        actor,
        torch.zeros(1, 60),
        args.output,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        reference = actor(sample).numpy()
    exported = session.run(["action"], {"obs": sample.numpy()})[0]
    max_error = float(abs(reference - exported).max())
    audit = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_iteration": int(checkpoint.get("iter", -1)),
        "actor_input": 60,
        "actor_output": 16,
        "actor_input_semantics": "57D official proprioception + delayed [b,H,D]",
        "fr_action_indices": [3, 4, 5],
        "onnx_max_abs_error": max_error,
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if max_error > 1.0e-5:
        raise RuntimeError(f"ONNX export mismatch: {max_error}")


if __name__ == "__main__":
    main()
