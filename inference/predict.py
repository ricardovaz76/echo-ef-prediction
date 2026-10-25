# Predicts EF from one A4C and one PSAX video of the same visit
#
# Run from the repo root:
#   python3 -m inference.predict --a4c path/to/a4c.avi --psax path/to/psax.avi \
#       --checkpoint checkpoints/best_model_mae.pth --output result.json

import json
import sys

from cli import parse_infer_args
from .predictor import EchoPredictor


def main(argv=None):
  args = parse_infer_args(argv)

  # Bad input (unreadable video, no cardiac motion, checkpoint without EF stats)
  # raises ValueError, shown as one line instead of a traceback
  try:
    predictor = EchoPredictor(args.checkpoint, device=args.device)
    ef = predictor.predict(args.a4c, args.psax)
  except ValueError as e:
    sys.exit(f"Error: {e}")

  print(f"Predicted EF: {ef:.1f}%")

  if args.output:
    result = {"a4c": args.a4c, "psax": args.psax, "checkpoint": args.checkpoint, "ef": ef}
    with open(args.output, "w") as f:
      json.dump(result, f, indent=2)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
  main()
