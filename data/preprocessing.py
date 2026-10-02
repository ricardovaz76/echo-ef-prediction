import os

import cv2
import numpy as np


# Reads a video and returns its frames as a numpy array of shape (T, 112, 112)
#
# Each frame is resized to 112x112, converted to grayscale and normalized to [0.0, 1.0].
# This is shared between training preprocessing and inference so both see the same input.
def load_video_frames(video_path):
    # Read each frame one-by-one
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize each frame to 112x112 and convert to Grayscale
        frame = cv2.resize(frame, (112, 112))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    cap.release()

    # Return an empty placeholder if no frames were grabbed
    if len(frames) == 0:
        return np.zeros((0, 112, 112), dtype=np.float32)

    # Stacks each frame into a numpy array of shape: (T, 112, 112)
    frames = np.array(frames)

    # Normalize frame pixels to [0.0, 1.0]
    return frames.astype(np.float32) / 255.0


# Frame extraction Method
#
# This method is used to extract all of the frames from each video of a given
# path before starting the training, this is to ensure training could just
# grab the frames rather than having to do the frame extraction process for
# each batch.
#
# I chose to extract all of the frames as opposed to a specific set to ensure
# the ED and ES frame was included in the training which is vital for the
# Ejection Fraction prediction
def extract_and_save(video_path, save_path):
    # Save frames into numpy path which is just the FileName.npy
    np.save(save_path, load_video_frames(video_path))


# Extracts the frames of every video in a view (A4C or PSAX) into save_dir as .npy files
# Only videos from patients in patient_ids are processed
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
