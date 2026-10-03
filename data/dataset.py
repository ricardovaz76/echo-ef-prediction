import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .masks import build_frame_dict, get_ed_es_by_area


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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # grab prebuilt sample pair
        a4c_fname, psax_fname, ef_val = self.samples[idx]

        # =========================================================
        # GRAB A4C VIDEO
        # =========================================================
        base_a4c = a4c_fname.replace(".avi", "")
        a4c = np.load(f"{self.a4c_dir}/{base_a4c}.npy")
        a4c = (torch.tensor(a4c, dtype=torch.float32) / 255.0).unsqueeze(1)  # (T,1,H,W), uint8 -> [0, 1]

        # =========================================================
        # GRAB PSAX VIDEO
        # =========================================================
        base_psax = psax_fname.replace(".avi", "")
        psax = np.load(f"{self.psax_dir}/{base_psax}.npy")
        psax = (torch.tensor(psax, dtype=torch.float32) / 255.0).unsqueeze(1)  # (T,1,H,W), uint8 -> [0, 1]

        # =========================================================
        # BUILD FULL SEGMENTATION VIDEOS
        # (annotated frames filled with masks, rest zeroed out)
        # =========================================================

        # Grab the total number of frames of a video [(T,1,H,W) T value]
        T_a4c = a4c.shape[0]
        T_psax = psax.shape[0]

        # Initialize a 0 matrix with the same dimensions as the a4c and psax videos
        seg_a4c = np.zeros((T_a4c, 1, 112, 112), dtype=np.float32)
        seg_psax = np.zeros((T_psax, 1, 112, 112), dtype=np.float32)

        # Stores only the annotated frames and leaves the rest of the frames as 0s
        # For both a4c and psax
        for (fname, frame), mask in self.mask_dict_a4c.items():
            if fname == a4c_fname and frame < T_a4c:
                seg_a4c[frame, 0] = mask

        for (fname, frame), mask in self.mask_dict_psax.items():
            if fname == psax_fname and frame < T_psax:
                seg_psax[frame, 0] = mask

        # converts annotated frames to torch
        seg_a4c = torch.from_numpy(seg_a4c).float()
        seg_psax = torch.from_numpy(seg_psax).float()

        # =========================================================
        # GRAB EJECTION FRACTIONS
        # =========================================================
        ef = torch.tensor(float(ef_val), dtype=torch.float32)

        # =========================================================
        # GRAB ED AND ES FRAME ANNOTATIONS
        # =========================================================
        ed_raw_a4c, es_raw_a4c = get_ed_es_by_area(
            self.mask_dict_a4c,
            a4c_fname,
            self.frame_dict_a4c[a4c_fname],
            max_frame=T_a4c
        )

        ed_raw_psax, es_raw_psax = get_ed_es_by_area(
            self.mask_dict_psax,
            psax_fname,
            self.frame_dict_psax[psax_fname],
            max_frame=T_psax
        )

        # safety: ED or ES frame falls back to 0 if no annotation is found
        ed_raw_a4c = 0 if ed_raw_a4c is None else ed_raw_a4c
        es_raw_a4c = 0 if es_raw_a4c is None else es_raw_a4c
        ed_raw_psax = 0 if ed_raw_psax is None else ed_raw_psax
        es_raw_psax = 0 if es_raw_psax is None else es_raw_psax

        # ---------------- OUTPUT ----------------
        return {
            # Raw video frames as tensors
            "a4c": a4c,
            "psax": psax,
            "ef": ef,

            # FULL segmentation videos
            "seg_a4c": seg_a4c,
            "seg_psax": seg_psax,

            # indices
            "ed_a4c": ed_raw_a4c,
            "es_a4c": es_raw_a4c,
            "ed_psax": ed_raw_psax,
            "es_psax": es_raw_psax,
        }
