import argparse


# Arguments for inference/predict.py
def parse_infer_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict ejection fraction from an A4C and a PSAX video")
    parser.add_argument("--a4c", required=True,
                        help="Path to the apical 4-chamber (A4C) video")
    parser.add_argument("--psax", required=True,
                        help="Path to the parasternal short-axis (PSAX) video of the same visit")
    parser.add_argument("--checkpoint", default="checkpoints/best_model_mae.pth",
                        help="Path to the model checkpoint (.pth)")
    parser.add_argument("--output", default=None,
                        help="Also save the full result as JSON to this path")
    parser.add_argument("--device", default=None,
                        help="Device to run on (e.g. cpu, cuda), defaults to cuda when available")
    return parser.parse_args(argv)
