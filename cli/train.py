import argparse

from data import CLIP_LENGTH, TARGET_FPS
from .common import add_data_args


# Arguments for training/train.py
def parse_train_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the EchoNet EF model")
    add_data_args(parser)
    parser.add_argument("--output-dir", default="checkpoints",
                        help="Folder to save checkpoints and training curves to")

    # Clip settings, saved in the checkpoint so testing and inference use the same ones
    clips = parser.add_argument_group("clips")
    clips.add_argument("--target-fps", type=int, default=TARGET_FPS,
                       help="Frame rate every video is resampled to")
    clips.add_argument("--clip-length", type=int, default=CLIP_LENGTH,
                       help="Frames per clip around ED and ES")

    # Phase 1: joint segmentation + EF regression
    phase1 = parser.add_argument_group("phase 1 (joint segmentation + EF)")
    phase1.add_argument("--lr-p1", type=float, default=1e-4,
                        help="Phase 1 learning rate")
    phase1.add_argument("--epochs-p1", type=int, default=50,
                        help="Maximum phase 1 epochs")
    phase1.add_argument("--dice-patience", type=int, default=5,
                        help="Phase 1 early stops after this many epochs without a better val Dice")

    # Phase 2: segmentation frozen, EF fine-tuning
    phase2 = parser.add_argument_group("phase 2 (EF fine-tuning)")
    phase2.add_argument("--lr-p2", type=float, default=5e-5,
                        help="Phase 2 starting learning rate")
    phase2.add_argument("--epochs-p2", type=int, default=50,
                        help="Maximum phase 2 epochs")
    phase2.add_argument("--mae-patience", type=int, default=7,
                        help="Phase 2 early stops after this many epochs without a better val MAE")
    phase2.add_argument("--sched-patience", type=int, default=3,
                        help="Epochs without a better val MAE before the learning rate is reduced")
    phase2.add_argument("--sched-factor", type=float, default=0.5,
                        help="Factor the learning rate is multiplied by when reduced")
    phase2.add_argument("--sched-min-lr", type=float, default=1e-6,
                        help="Lowest the learning rate is reduced to")

    return parser.parse_args(argv)
