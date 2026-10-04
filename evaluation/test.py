# Evaluates a trained EchoNetModel checkpoint on the test split
#
# Prints the test metrics and saves the segmentation overlays, regression,
# ROC, and Bland-Altman plots to --output-dir. With --video, also saves the
# A4C and PSAX segmentation videos for one test sample.
#
# Run from the repo root:
#   python -m evaluation.test --dataset-root path/to/dataset --processed-root path/to/processed \
#       --checkpoint checkpoints/best_model_mae.pth

import os

import torch

from cli import parse_test_args
from data import CLIP_LENGTH, TARGET_FPS, build_dataloaders, build_datasets
from models import build_model
from .evaluate import evaluate, print_metrics
from .plots import plot_bland_altman, plot_regression, plot_roc, plot_segmentation_overlays
from .video import render_sample_videos


# The model predicts normalized EF, so the train set stats are needed to convert back to EF %
# Uses --ef-mean/--ef-std if given, otherwise the values saved in the checkpoint
def resolve_ef_stats(args, checkpoint):
    ef_mean = args.ef_mean if args.ef_mean is not None else checkpoint.get("ef_mean")
    ef_std  = args.ef_std  if args.ef_std  is not None else checkpoint.get("ef_std")

    if ef_mean is None or ef_std is None:
        raise SystemExit(
            "This checkpoint doesn't store ef_mean/ef_std. "
            "Pass the train set values with --ef-mean and --ef-std."
        )
    return ef_mean, ef_std


# Clip settings the model was trained with, so test clips are built the same way
# Checkpoints saved before these were stored fall back to the defaults
def resolve_clip_config(checkpoint):
    return checkpoint.get("config", {"clip_length": CLIP_LENGTH, "target_fps": TARGET_FPS})


def main(argv=None):
    args = parse_test_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())

    # -------------------------
    # model
    # -------------------------
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    ef_mean, ef_std = resolve_ef_stats(args, checkpoint)
    config = resolve_clip_config(checkpoint)

    model = build_model(device, clip_length=config["clip_length"])
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded {args.checkpoint} (EF_MEAN: {ef_mean}, EF_STD: {ef_std}, clips: {config})")

    # -------------------------
    # data
    # -------------------------
    train_dataset, val_dataset, test_dataset = build_datasets(
        args.dataset_root, args.processed_root, extract_frames=not args.skip_extraction, **config
    )
    _, _, test_loader = build_dataloaders(
        train_dataset, val_dataset, test_dataset, num_workers=args.num_workers
    )
    print(f"Test Dataset Pairs: {len(test_dataset)}")

    # -------------------------
    # evaluate
    # -------------------------
    metrics, all_preds, all_targets, all_samples = evaluate(
        model, test_loader, device, ef_mean, ef_std
    )
    print_metrics(metrics)

    # -------------------------
    # plots
    # -------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    plot_segmentation_overlays(all_samples, os.path.join(args.output_dir, "overlays"), num_samples=args.num_overlays)
    plot_regression(all_preds, all_targets, args.output_dir)
    roc_auc = plot_roc(all_preds, all_targets, os.path.join(args.output_dir, "roc.png"))
    plot_bland_altman(all_preds, all_targets, os.path.join(args.output_dir, "bland_altman.png"))

    print(f"ROC AUC (EF < 50):   {roc_auc:.4f}")
    print(f"Saved plots to {args.output_dir}")

    # -------------------------
    # segmentation videos
    # -------------------------
    if args.video:
        video_dir = os.path.join(args.output_dir, "videos")
        render_sample_videos(model, test_dataset, device, ef_mean, ef_std, video_dir, seed=args.video_seed)
        print(f"Saved segmentation videos to {video_dir}")


if __name__ == "__main__":
    main()
