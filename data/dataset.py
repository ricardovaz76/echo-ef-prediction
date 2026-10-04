import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .masks import build_frame_dict, get_ed_es_by_area
from .preprocessing import clip_path


# Grabs the visit ID to ensure the video pairs of each
# patient is paired by Visit IDs
def get_visit_id(filename):
    return filename.split("-")[1]


class EchoDataset(Dataset):

    def __init__(self, a4c_df, psax_df, a4c_dir, psax_dir, patient_ids, mask_dict_a4c, mask_dict_psax):

        self.a4c_df = a4c_df.set_index("patient_id")
        self.psax_df = psax_df.set_index("patient_id")

        self.patients = list(patient_ids)

        self.a4c_dir = a4c_dir
        self.psax_dir = psax_dir

        self.mask_dict_a4c = mask_dict_a4c
        self.mask_dict_psax = mask_dict_psax

        self.frame_dict_a4c = build_frame_dict(mask_dict_a4c)
        self.frame_dict_psax = build_frame_dict(mask_dict_psax)

        # Grabs every video pair for each patient ensuring each video is from the same visit.
        self.samples = []
        for pid in self.patients:
            a4c_rows = self.a4c_df.loc[pid]
            psax_rows = self.psax_df.loc[pid]

            if isinstance(a4c_rows, pd.Series):
                a4c_rows = a4c_rows.to_frame().T
            if isinstance(psax_rows, pd.Series):
                psax_rows = psax_rows.to_frame().T

            for _, a4c_row in a4c_rows.drop_duplicates(subset="FileName").iterrows():
                for _, psax_row in psax_rows.drop_duplicates(subset="FileName").iterrows():
                    if get_visit_id(a4c_row.FileName) == get_visit_id(psax_row.FileName):
                        self.samples.append((
                            a4c_row.FileName,
                            psax_row.FileName,
                            float(a4c_row.EF)
                        ))

        # Keep only pairs where both views have a saved clip. Extraction skips videos
        # without both an ED and ES tracing or with ED and ES too far apart.
        self.samples = [
            (a4c_fname, psax_fname, ef)
            for a4c_fname, psax_fname, ef in self.samples
            if os.path.exists(clip_path(self.a4c_dir, a4c_fname))
            and os.path.exists(clip_path(self.psax_dir, psax_fname))
        ]

    def __len__(self):
        return len(self.samples)

    # Loads one view's clip and builds its segmentation targets
    #
    # Returns:
    #   video:  (T, 1, H, W) frames normalized to [0, 1]
    #   seg:    (T, 1, H, W) ground truth masks on the annotated frames, zeros elsewhere
    #   ed, es: clip positions of the ED and ES frames
    #   n_valid: number of real frames, the rest are padding
    def _load_view(self, fname, clip_dir, mask_dict, frame_dict):
        path = clip_path(clip_dir, fname)
        data = np.load(path)

        # Clips saved before n_valid was added can't tell real frames from padding
        if "n_valid" not in data.files:
            raise KeyError(f"{path} has no n_valid, re-run extraction without --skip-extraction")

        frames, clip_src, n_valid = data["frames"], data["clip_src"], int(data["n_valid"])

        video = (torch.tensor(frames, dtype=torch.float32) / 255.0).unsqueeze(1)   # uint8 -> [0, 1]

        # Each annotated frame's mask goes on the clip frame it came from. Padding can
        # repeat the last frame, so only the first copy gets the mask to avoid counting
        # the same tracing more than once in the segmentation loss.
        seg = np.zeros((len(frames), 1, 112, 112), dtype=np.float32)
        clip_frames = []
        for frame in frame_dict[fname]:
            positions = np.flatnonzero(clip_src == frame)
            if len(positions) == 0:
                continue
            seg[positions[0], 0] = mask_dict[(fname, frame)]
            clip_frames.append(frame)

        # ED and ES from the annotated frames in the clip, as clip positions
        ed, es = get_ed_es_by_area(mask_dict, fname, clip_frames)

        # safety: ED or ES frame falls back to 0 if no annotation is found
        ed = 0 if ed is None else int(np.flatnonzero(clip_src == ed)[0])
        es = 0 if es is None else int(np.flatnonzero(clip_src == es)[0])

        return video, torch.from_numpy(seg), ed, es, n_valid

    def __getitem__(self, idx):

        # grab prebuilt sample pair
        a4c_fname, psax_fname, ef_val = self.samples[idx]

        a4c, seg_a4c, ed_a4c, es_a4c, valid_a4c = self._load_view(
            a4c_fname, self.a4c_dir, self.mask_dict_a4c, self.frame_dict_a4c
        )
        psax, seg_psax, ed_psax, es_psax, valid_psax = self._load_view(
            psax_fname, self.psax_dir, self.mask_dict_psax, self.frame_dict_psax
        )

        ef = torch.tensor(float(ef_val), dtype=torch.float32)

        # ---------------- OUTPUT ----------------
        return {
            # Clip frames as tensors
            "a4c": a4c,
            "psax": psax,
            "ef": ef,

            # FULL segmentation videos
            "seg_a4c": seg_a4c,
            "seg_psax": seg_psax,

            # clip positions of ED and ES
            "ed_a4c": ed_a4c,
            "es_a4c": es_a4c,
            "ed_psax": ed_psax,
            "es_psax": es_psax,

            # number of real frames, the rest of the clip is padding
            "valid_a4c": valid_a4c,
            "valid_psax": valid_psax,
        }
