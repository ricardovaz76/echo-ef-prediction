import os

import cv2
import numpy as np


def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (112, 112))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames.append(frame)

    cap.release()

    # Return an empty placeholder if no frames were grabbed
    if len(frames) == 0:
        return np.zeros((0, 112, 112), dtype=np.uint8)

    # Kept as raw uint8 pixels (0-255) to save 4x disk space over float32,
    # normalized to [0.0, 1.0] when loaded in EchoDataset
    return np.array(frames, dtype=np.uint8)    # (T, 112, 112)


# Saves frames into save_path to train without needing to extract frames for every epoch
def extract_and_save(video_path, save_path):
    np.save(save_path, load_video_frames(video_path))


# Creates a save path for each video before extraction
def preprocess_view(df, video_dir, save_dir, patient_ids):
    os.makedirs(save_dir, exist_ok=True)

    fname_to_pid = df.drop_duplicates(subset="FileName").set_index("FileName")["patient_id"].to_dict()

    for fname in df["FileName"].unique():

        if fname_to_pid[fname] not in patient_ids:
            continue

        video_path = os.path.join(video_dir, fname)

        save_path = os.path.join(
            save_dir,
            fname.replace(".avi", ".npy")
        )

        extract_and_save(video_path, save_path)
