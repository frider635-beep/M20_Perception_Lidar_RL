#!/usr/bin/env python3
"""Export the two deployable HR Bridge actors from a runner checkpoint."""

import argparse
import json
import os

import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()


def mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(), nn.Linear(128, output_dim),
    )


def load_actor(state, prefix, input_dim, output_dim):
    actor = mlp(input_dim, output_dim).eval()
    actor.load_state_dict({
        key.removeprefix(prefix + ".mlp."): value
        for key, value in state.items() if key.startswith(prefix + ".mlp.")
    })
    return actor


def main():
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = loaded["model_state_dict"]
    base = load_actor(state, "base_actor", 57, 16)
    bridge = load_actor(state, "bridge_actor", 372, 3)
    os.makedirs(args.output, exist_ok=True)
    base_path = os.path.join(args.output, "base_actor_57x16.onnx")
    bridge_path = os.path.join(args.output, "hr_bridge_actor_372x3.onnx")
    torch.onnx.export(
        base, torch.zeros(1, 57), base_path,
        input_names=["proprioception_57"], output_names=["base_action_16"],
        dynamic_axes={"proprioception_57": {0: "batch"}, "base_action_16": {0: "batch"}},
        opset_version=17,
    )
    torch.onnx.export(
        bridge, torch.zeros(1, 372), bridge_path,
        input_names=["proprioception_57_plus_elevation_315"], output_names=["hr_action_3"],
        dynamic_axes={"proprioception_57_plus_elevation_315": {0: "batch"}, "hr_action_3": {0: "batch"}},
        opset_version=17,
    )
    audit = {
        "base_input": 57, "base_output": 16,
        "bridge_input": 372, "bridge_output": 3,
        "bridge_input_semantics": "57D proprioception + 315D PGTT elevation",
        "simulator_hr_z_in_actor_input": False,
    }
    with open(os.path.join(args.output, "onnx_interface_audit.json"), "w") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
