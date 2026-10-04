# Predicts ejection fraction from an A4C and a PSAX video of the same visit
#
# Prints the EF and the heartbeats found in each video, and with --output saves
# the full result (including the area curves) as JSON.
#
# Run from the repo root:
#   python3 -m inference.predict --a4c path/to/a4c.avi --psax path/to/psax.avi

import json
import os
import sys

from cli import parse_infer_args
from .predictor import EchoPredictor


def print_result(result):
    print(f"\nPredicted EF: {result['ef']:.1f}%")

    efs = [b["ef"] for b in result["beat_efs"]]
    if len(efs) > 1:
        print(f"  averaged over {len(efs)} beat combinations (range {min(efs):.1f}-{max(efs):.1f}%)")

    for view in ("a4c", "psax"):
        v = result[view]
        print(f"\n{view.upper()}: {v['duration']:.2f}s at {v['fps']:.0f} fps, {len(v['beats'])} beat(s)")
        for i, b in enumerate(v["beats"]):
            note = "  (fallback: no clear beat, low confidence)" if b["fallback"] else ""
            print(f"  beat {i + 1}: ED {b['ed_time']:.2f}s (frame {b['ed_frame']}) -> ES {b['es_time']:.2f}s (frame {b['es_frame']}){note}")


def main(argv=None):
    args = parse_infer_args(argv)

    predictor = EchoPredictor(args.checkpoint, device=args.device)

    try:
        result = predictor.predict(args.a4c, args.psax)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    print_result(result)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved full result to {args.output}")


if __name__ == "__main__":
    main()
