import os

import pandas as pd


# Split folds defined in FileList.csv
TRAIN_SPLITS = [0, 1, 2, 3, 4, 5, 6, 7]
VAL_SPLITS   = [8]
TEST_SPLITS  = [9]


# Extracts the patient ID from the FileName (the part before the first "-")
def add_patient_id(df):
    df = df.copy()
    df["patient_id"] = df["FileName"].apply(lambda x: x.split("-")[0])
    return df


# Grabs the file list and volume tracings for a single view (A4C or PSAX)
# view_dir is the folder containing FileList.csv and VolumeTracings.csv
def load_view(view_dir):
    df = pd.read_csv(os.path.join(view_dir, "FileList.csv"))
    df_volume = pd.read_csv(os.path.join(view_dir, "VolumeTracings.csv"))

    return add_patient_id(df), add_patient_id(df_volume)


# Attaches the Split and EF labels from the file list to every volume tracing row
def merge_labels(df_volume, df):
    return df_volume.merge(
        df[["FileName", "Split", "EF"]],
        on="FileName",
        how="left"
    )


# Keeps only the rows that belong to the given split folds
def split_by_fold(df, splits):
    return df[df["Split"].isin(splits)]


# Filters out any invalid frames and converts them to 0-based integers
# to prevent a frame from going out of bounds when extracting
def clean_frames(df_volume):
    df_volume = df_volume[pd.to_numeric(df_volume["Frame"], errors="coerce").notnull()].copy()
    df_volume["Frame"] = df_volume["Frame"].astype(int) - 1
    return df_volume


# Patients that have both an A4C and a PSAX video
def get_common_patients(a4c_df, psax_df):
    return set(a4c_df["patient_id"].unique()) & set(psax_df["patient_id"].unique())
