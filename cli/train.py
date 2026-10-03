import argparse

from .common import add_data_args


# Arguments for training/train.py
def parse_train_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the EchoNet EF model")
    add_data_args(parser)
    parser.add_argument("--output-dir", default="checkpoints",
                        help="Folder to save checkpoints and training curves to")
    return parser.parse_args(argv)
