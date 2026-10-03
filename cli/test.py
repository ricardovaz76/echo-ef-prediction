import argparse

from .common import add_data_args


# Arguments for evaluation/test.py
def parse_test_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the EchoNet EF model on the test split")
    add_data_args(parser)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the model checkpoint (.pth)")
    parser.add_argument("--output-dir", default="outputs/test",
                        help="Folder to save the plots to")
    parser.add_argument("--num-overlays", type=int, default=10,
                        help="Number of segmentation overlay figures to save")
    parser.add_argument("--video", action="store_true",
                        help="Also save A4C and PSAX segmentation videos for one test sample")
    parser.add_argument("--video-seed", type=int, default=None,
                        help="Seed for picking the video sample, random if not set")
    parser.add_argument("--ef-mean", type=float, default=None,
                        help="Train set EF mean, for checkpoints that don't store it")
    parser.add_argument("--ef-std", type=float, default=None,
                        help="Train set EF std, for checkpoints that don't store it")
    return parser.parse_args(argv)
