import json
import os

from .clips import CLIP_LENGTH
from .dataset import EchoDataset
from .loading import (
    TEST_SPLITS,
    TRAIN_SPLITS,
    VAL_SPLITS,
    clean_frames,
    get_common_patients,
    load_view,
    merge_labels,
    split_by_fold,
)
from .masks import build_mask_dict
from .preprocessing import TARGET_FPS, preprocess_view


# Records the clip settings next to the extracted clips, so clips can't be
# reused with settings they weren't made with
def clip_config_path(processed_root):
    return os.path.join(processed_root, "clip_config.json")


def check_clip_config(processed_root, clip_config):
    path = clip_config_path(processed_root)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No {path}, re-run extraction without --skip-extraction")

    with open(path) as f:
        saved = json.load(f)
    if saved != clip_config:
        raise ValueError(
            f"Clips in {processed_root} were extracted with {saved}, not {clip_config}. "
            "Re-run extraction without --skip-extraction"
        )


# Runs the full data pipeline and returns the train/val/test datasets
#
# Follows the same order as notebooks/data_exploration.ipynb, which runs these
# steps one at a time for inspection.
#
# dataset_root:   folder containing the A4C/ and PSAX/ folders
# processed_root: folder to save the extracted clips (.npz) to
# extract_frames: set to False to reuse clips already extracted to processed_root
# clip_length, target_fps: clip settings, must match the ones the clips were extracted with
def build_datasets(dataset_root, processed_root, extract_frames=True, clip_length=CLIP_LENGTH, target_fps=TARGET_FPS):
    clip_config = {"clip_length": clip_length, "target_fps": target_fps}

    a4c_dir  = os.path.join(dataset_root, "A4C")
    psax_dir = os.path.join(dataset_root, "PSAX")

    processed_a4c  = os.path.join(processed_root, "a4c")
    processed_psax = os.path.join(processed_root, "psax")

    # Load the file lists and volume tracings, then attach Split and EF to each tracing
    df_a4c, df_a4c_volume = load_view(a4c_dir)
    df_psax, df_psax_volume = load_view(psax_dir)

    df_a4c_volume = merge_labels(df_a4c_volume, df_a4c)
    df_psax_volume = merge_labels(df_psax_volume, df_psax)

    # Split before cleaning frames
    train_a4c_volume = split_by_fold(df_a4c_volume, TRAIN_SPLITS)
    val_a4c_volume   = split_by_fold(df_a4c_volume, VAL_SPLITS)
    test_a4c_volume  = split_by_fold(df_a4c_volume, TEST_SPLITS)

    train_psax_volume = split_by_fold(df_psax_volume, TRAIN_SPLITS)
    val_psax_volume   = split_by_fold(df_psax_volume, VAL_SPLITS)
    test_psax_volume  = split_by_fold(df_psax_volume, TEST_SPLITS)

    # Patients with both an A4C and PSAX video in each split
    train_patients = get_common_patients(train_a4c_volume, train_psax_volume)
    val_patients   = get_common_patients(val_a4c_volume, val_psax_volume)
    test_patients  = get_common_patients(test_a4c_volume, test_psax_volume)

    common_patients_all = train_patients | val_patients | test_patients

    # Ground truth segmentation masks
    mask_dict_a4c = build_mask_dict(clean_frames(df_a4c_volume))
    mask_dict_psax = build_mask_dict(clean_frames(df_psax_volume))

    # Clip extraction
    if extract_frames:
        preprocess_view(df_a4c, os.path.join(a4c_dir, "Videos"), processed_a4c, common_patients_all, mask_dict_a4c, clip_length, target_fps)
        preprocess_view(df_psax, os.path.join(psax_dir, "Videos"), processed_psax, common_patients_all, mask_dict_psax, clip_length, target_fps)

        with open(clip_config_path(processed_root), "w") as f:
            json.dump(clip_config, f)
    else:
        check_clip_config(processed_root, clip_config)

    train_dataset = EchoDataset(train_a4c_volume, train_psax_volume, processed_a4c, processed_psax, train_patients, mask_dict_a4c, mask_dict_psax)
    val_dataset   = EchoDataset(val_a4c_volume, val_psax_volume, processed_a4c, processed_psax, val_patients, mask_dict_a4c, mask_dict_psax)
    test_dataset  = EchoDataset(test_a4c_volume, test_psax_volume, processed_a4c, processed_psax, test_patients, mask_dict_a4c, mask_dict_psax)

    return train_dataset, val_dataset, test_dataset
