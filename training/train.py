# Two-phase training for EchoNetModel
#
# Phase 1: joint segmentation + EF regression, keeps the best val Dice checkpoint
# Phase 2: segmentation (including the shared encoder) frozen, fine-tunes the
#          EF regression and keeps the best val MAE checkpoint
#
# Run from the repo root:
#   python -m training.train --dataset-root path/to/dataset --processed-root path/to/processed

import os

import matplotlib.pyplot as plt
import torch

from cli import parse_train_args
from data import build_dataloaders, build_datasets, compute_ef_stats
from models import build_model
from .engine import train_one_epoch, validate


def plot_training_curves(train_losses, val_maes, val_dices, phase2_start, path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

    # Training loss
    ax1.plot(train_losses, label="Train Loss")
    ax1.axvline(x=phase2_start, color='gray', linestyle='--', label="Phase 2 start")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Val MAE
    ax2.plot(val_maes, label="Val MAE", color="orange")
    ax2.axvline(x=phase2_start, color='gray', linestyle='--', label="Phase 2 start")
    ax2.set_title("Validation MAE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.legend()

    # Val Dice
    ax3.plot(val_dices, label="Val Dice", color="green")
    ax3.axvline(x=phase2_start, color='gray', linestyle='--', label="Phase 2 start")
    ax3.set_title("Validation Dice")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Dice")
    ax3.legend()

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    args = parse_train_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())

    os.makedirs(args.output_dir, exist_ok=True)
    dice_ckpt_path = os.path.join(args.output_dir, "best_model_dice.pth")
    mae_ckpt_path  = os.path.join(args.output_dir, "best_model_mae.pth")

    # -------------------------
    # data
    # -------------------------
    train_dataset, val_dataset, test_dataset = build_datasets(
        args.dataset_root, args.processed_root, extract_frames=not args.skip_extraction
    )
    train_loader, val_loader, _ = build_dataloaders(
        train_dataset, val_dataset, test_dataset, num_workers=args.num_workers
    )

    print(f"Train Dataset Pairs: {len(train_dataset)}")
    print(f"Validation Dataset Pairs: {len(val_dataset)}")

    ef_mean, ef_std = compute_ef_stats(train_dataset)
    print("EF_MEAN:", ef_mean)
    print("EF_STD:", ef_std)

    # -------------------------
    # model
    # -------------------------
    model = build_model(device)

    # -------------------------
    # history
    # -------------------------
    train_losses     = []
    val_maes         = []
    val_dices        = []

    # =========================================================
    # PHASE 1: joint training, save best dice
    # =========================================================
    print("=" * 50)
    print("PHASE 1: joint seg + reg training")
    print("=" * 50)

    # Adam optimizer for Phase 1 training
    optimizer_p1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_p1,
    )

    best_val_dice = 0.0
    dice_counter  = 0

    for epoch in range(args.epochs_p1):

        # Train the epoch
        loss, reg_loss, seg_loss = train_one_epoch(
            model, train_loader, optimizer_p1, device, ef_mean, ef_std, seg_enable=True
        )

        # Validate the epoch performance
        metrics = validate(model, val_loader, device, ef_mean, ef_std)
        val_mae  = metrics["mae"]
        val_dice = metrics["dice"]

        # Save losses and val metrics for loss and val curve
        train_losses.append(loss)
        val_maes.append(val_mae)
        val_dices.append(val_dice)

        # Save the best Dice model
        # Early stop if Dice stops improving
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            dice_counter  = 0
            torch.save({
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer_p1.state_dict(),
                "epoch":                epoch,
                "val_mae":              float(val_mae),
                "val_dice":             float(val_dice),
                "ef_mean":              ef_mean,
                "ef_std":               ef_std,
            }, dice_ckpt_path)
            print(f"    Saved best dice model (dice={best_val_dice:.4f})")
        else:
            dice_counter += 1
            if dice_counter >= args.dice_patience:
                print(f"  Early stopping phase 1 at epoch {epoch+1}")
                break

        # Print results for Phase 1
        print(f"[Phase 1 | Epoch {epoch+1}]")
        print(f"  Total Loss: {loss:.4f}  Reg: {reg_loss:.4f}  Seg: {seg_loss:.4f}")
        print(f"  Val MAE:    {val_mae:.4f}  Val Dice: {val_dice:.4f}")
        print(f"  R²: {metrics['r2']:.4f}  Within 5: {metrics['within_5']:.4f}")
        print("-" * 50)

    # Phase 2 starts after however many epochs phase 1 actually ran
    phase2_start = len(train_losses)

    # =========================================================
    # PHASE 2: freeze seg, train reg only, save best MAE
    # =========================================================
    print("\n" + "=" * 50)
    print("PHASE 2: seg frozen, reg fine-tuning")
    print("=" * 50)

    # load best dice checkpoint before freezing
    checkpoint = torch.load(dice_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded best dice model from epoch {checkpoint['epoch']+1}")

    # freeze segmentation (includes the shared encoder)
    for param in model.seg_model.parameters():
        param.requires_grad = False

    # Fresh Adam optimizer for Phase 2 MAE fine-tuning (Excludes frozen segmentation)
    optimizer_p2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_p2,
    )

    # Scheduler:
    # Looks for improved MAE score, if the score does not improve
    # after 3 epochs, then the learning rate is halved
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p2,
        mode="min",
        factor=args.sched_factor,
        patience=args.sched_patience,
        min_lr=args.sched_min_lr
    )

    best_val_mae = float("inf")
    mae_counter  = 0

    for epoch in range(args.epochs_p2):

        # Train the epoch
        loss, reg_loss, seg_loss = train_one_epoch(
            model, train_loader, optimizer_p2, device, ef_mean, ef_std, seg_enable=False
        )

        # Validate the epoch performance
        metrics = validate(model, val_loader, device, ef_mean, ef_std)
        val_mae  = metrics["mae"]
        val_dice = metrics["dice"]

        # Updates scheduler
        # If mae improves then scheduler resets
        # If mae does not improve for 3 epochs then schduler halves the learning rate
        scheduler.step(val_mae)

        # Save losses and val metrics for loss and val curve
        train_losses.append(loss)
        val_maes.append(val_mae)
        val_dices.append(val_dice)

        # Save the best MAE model
        # Early stop if MAE stops improving
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            mae_counter  = 0
            torch.save({
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer_p2.state_dict(),
                "epoch":                epoch,
                "val_mae":              float(val_mae),
                "val_dice":             float(val_dice),
                "ef_mean":              ef_mean,
                "ef_std":               ef_std,
            }, mae_ckpt_path)
            print(f"    Saved best MAE model (mae={best_val_mae:.4f})")
        else:
            mae_counter += 1
            if mae_counter >= args.mae_patience:
                print(f"  Early stopping phase 2 at epoch {epoch+1}")
                break

        # Print the Phase 2 Results
        print(f"[Phase 2 | Epoch {epoch+1}]")
        print(f"  Total Loss: {loss:.4f}  Reg: {reg_loss:.4f}  Seg: {seg_loss:.4f}")
        print(f"  Val MAE:    {val_mae:.4f}  Val Dice: {val_dice:.4f}")
        print(f"  R²: {metrics['r2']:.4f}  Within 5: {metrics['within_5']:.4f}")
        print("-" * 50)

    # Print the full training results
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print(f"  Best Dice: {best_val_dice:.4f}")
    print(f"  Best MAE:  {best_val_mae:.4f}")
    print("=" * 50)

    # Loss, Validation, and Dice Curves
    curves_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_curves(train_losses, val_maes, val_dices, phase2_start, curves_path)
    print(f"Saved training curves to {curves_path}")


if __name__ == "__main__":
    main()
