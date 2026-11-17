"""Search for an official-provenance M20 checkpoint without loading any model."""

from pathlib import Path


ROOTS = (
    Path("/home/ubuntu/m20_stair_official20k"),
    Path("/home/ubuntu/M20 PRO RL"),
    Path("/home/ubuntu/m20_perceptive_stairs"),
    Path("/home/ubuntu/m20_perceptive_final"),
)


def main():
    candidates = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".onnx"}:
                candidates.append(path)
    print("CHECKPOINT_SEARCH_ROOTS=" + ";".join(str(root) for root in ROOTS))
    print("CHECKPOINT_CANDIDATE_COUNT=" + str(len(candidates)))
    for path in candidates[:40]:
        print("EXCLUDED_CANDIDATE=" + str(path))
    print("OFFICIAL_PRETRAINED_CHECKPOINT_NOT_FOUND")
    print("DIAGNOSTIC_POLICY_LOADED=False")
    print("FINAL_POLICY_INITIALIZATION_UNCHANGED=FROM_SCRATCH")


if __name__ == "__main__":
    main()
