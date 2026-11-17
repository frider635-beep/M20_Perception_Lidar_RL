#!/usr/bin/env python3
"""Static and ONNX architecture gates for M20 V10."""

from __future__ import annotations

import os
import tempfile
import importlib.util
from pathlib import Path

import numpy as np
import torch

POLICY_PATH = Path(__file__).resolve().parents[2] / "source/rl_training/rl_training/v10_unified_policy.py"
SPEC = importlib.util.spec_from_file_location("m20_v10_policy_standalone", POLICY_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
UnifiedActor = MODULE.UnifiedActor


FR = list(range(3, 6))
HR = list(range(9, 12))
SUPPORT = list(range(0, 3)) + list(range(6, 16))


def main():
    torch.manual_seed(10)
    actor = UnifiedActor().eval()
    first = torch.randn(8, 60)
    first[:, 57] = 1.0
    second = first.clone()
    first[:, 58:60] = torch.tensor([0.05, 0.30])
    second[:, 58:60] = torch.tensor([0.20, 0.15])
    with torch.inference_mode():
        out_a, out_b = actor(first), actor(second)
    support_diff = (out_a[:, SUPPORT] - out_b[:, SUPPORT]).abs().max().item()
    fr_diff = (out_a[:, FR] - out_b[:, FR]).abs().max().item()
    hr_diff = (out_a[:, HR] - out_b[:, HR]).abs().max().item()
    b_changed = first.clone()
    b_changed[:, 57] = 0.0
    with torch.inference_mode():
        b_diff = (out_a - actor(b_changed)).abs().max().item()

    print(f"TEST_A_ACTOR_INPUT={first.shape[-1]}")
    print(f"TEST_A_ACTOR_OUTPUT={out_a.shape[-1]}")
    print(f"HD_TO_SUPPORT_MAX_DIFF={support_diff:.12e}")
    print(f"HD_TO_FR_MAX_DIFF={fr_diff:.12e}")
    print(f"HD_TO_HR_MAX_DIFF={hr_diff:.12e}")
    print(f"B_TO_POLICY_MAX_DIFF={b_diff:.12e}")
    print(f"TEST_A={'PASS' if first.shape[-1] == 60 and out_a.shape[-1] == 16 else 'FAIL'}")
    direct_ok = support_diff <= 1e-12 and fr_diff > 0 and hr_diff <= 1e-12
    print(f"TEST_B={'PASS' if direct_ok else 'FAIL'}")
    print(f"TEST_C={'PASS' if b_diff > 0 else 'FAIL'}")

    onnx_status = "FAIL"
    onnx_support_diff = float("inf")
    try:
        import onnx
        from onnx.reference import ReferenceEvaluator

        with tempfile.TemporaryDirectory(prefix="m20_v10_onnx_") as directory:
            path = os.path.join(directory, "actor.onnx")
            torch.onnx.export(
                actor, first[:1], path, opset_version=18,
                input_names=["obs"], output_names=["actions"],
                dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            )
            session = ReferenceEvaluator(onnx.load(path))
            ort_a = session.run(None, {"obs": first.numpy()})[0]
            ort_b = session.run(None, {"obs": second.numpy()})[0]
            onnx_support_diff = float(np.max(np.abs(ort_a[:, SUPPORT] - ort_b[:, SUPPORT])))
            onnx_fr_diff = float(np.max(np.abs(ort_a[:, FR] - ort_b[:, FR])))
            onnx_hr_diff = float(np.max(np.abs(ort_a[:, HR] - ort_b[:, HR])))
            onnx_status = (
                "PASS" if onnx_support_diff <= 1e-7 and onnx_fr_diff > 0 and onnx_hr_diff <= 1e-7
                else "FAIL"
            )
    except Exception as error:
        print(f"ONNX_ERROR={type(error).__name__}:{error}")
    print(f"ONNX_HD_TO_SUPPORT_MAX_DIFF={onnx_support_diff:.12e}")
    print(f"TEST_E={onnx_status}")
    # Force recognizable head outputs and verify exact 16D reassembly order.
    class ConstantHead(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

        def forward(self, value):
            return self.values.expand(value.shape[0], -1)

    ordered = UnifiedActor().eval()
    ordered.shared_trunk = ConstantHead([0.0] * 128)
    ordered.support_head = ConstantHead(list(range(100, 113)))
    ordered.fr_head = ConstantHead([300, 301, 302])
    ordered_output = ordered(torch.zeros(1, 60))[0]
    expected = torch.tensor(
        [100, 101, 102, 300, 301, 302, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        dtype=torch.float32,
    )
    order_ok = torch.equal(ordered_output, expected)
    passed = first.shape[-1] == 60 and out_a.shape[-1] == 16 and direct_ok and b_diff > 0 and order_ok
    print(f"V10_UNIFIED_ARCHITECTURE={'PASS' if passed else 'FAIL'}")
    print(f"FR_SPECIAL_HEAD_MAPPING={'PASS' if order_ok else 'FAIL'}")
    print(f"HD_TO_FR_ONLY={'PASS' if direct_ok else 'FAIL'}")
    print(f"HD_TO_HR_DIRECT_PATH={'FALSE' if hr_diff <= 1e-12 else 'TRUE'}")
    print(f"ACTION_16_ORDER_PRESERVED={'PASS' if order_ok else 'FAIL'}")
    raise SystemExit(0 if passed and onnx_status == "PASS" else 1)


if __name__ == "__main__":
    main()
