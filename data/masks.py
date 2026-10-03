from collections import defaultdict

import cv2
import numpy as np


# Creates the ground truth masks from given contours
def create_mask(coords, size=112):

    # Empty 112x112 matrix
    mask = np.zeros((size, size), dtype=np.uint8)

    # No coordinates were passed to generate the mask
    if coords is None or len(coords) == 0:
        return mask

    # Filter out any invalid coordinates
    coords = [(x, y) for x, y in coords if not (np.isnan(x) or np.isnan(y))]

    # Avoid creating mask if less than 3 coordinates were found
    if len(coords) < 3:
        return mask
    if len(set(coords)) < 3:
        return mask

    # Generate Binary mask by filling in the contour space
    pts = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask


# Gathers all of the coordinates of the FileName and frame number pair and creates
# the segmentation mask and stores them in the dictionary
#
# Key: (FileName, Frame)
#
# Value: Binary mask
def build_mask_dict(df_volume):
    mask_dict = {}
    for (fname, frame), group in df_volume.groupby(["FileName", "Frame"]):
        coords = list(zip(group["X"], group["Y"]))
        mask_dict[(fname, frame)] = create_mask(coords)
    return mask_dict


# Lookup Dictionary that maps each filename to a list of all the frame numbers
# that have annotations for it.
#
# Key: FileName
#
# Value: List of frame numbers
def build_frame_dict(mask_dict):
    frame_dict = defaultdict(list)
    for (fname, frame) in mask_dict.keys():
        frame_dict[fname].append(frame)
    return frame_dict


# Grabs the ED and ES contours based on their areas
# Largest area is a ED contour
# Smallest area is a ES contour
def get_ed_es_by_area(mask_dict, file_name, frame_list, max_frame=None):
    # Return if the frame list is empty (No frames found)
    if len(frame_list) == 0:
        return None, None
    # Drops any frames that are out of bounds of actual max frames
    if max_frame is not None:
        frame_list = [f for f in frame_list if f < max_frame]

    # Return if no frames are found
    if len(frame_list) == 0:
        return None, None

    # Grabs the sum of each mask and stores it in the areas list
    # Each mask is a binary mask meaning the foreground is 1s while background is 0s
    # This means if a sum of a mask is a large number than its most likely an ED frame
    # If the sum of a mask is a smaller number than its most likely an ES frame
    areas = []
    for f in frame_list:
        mask = mask_dict.get((file_name, f), None)
        if mask is None:
            continue
        areas.append((f, mask.sum()))

    # Return nothing if no areas were stored
    if len(areas) == 0:
        return None, None

    # Filter out any negative or 0 area values, return if nothing was found
    areas = [(f, a) for f, a in areas if a > 0]
    if len(areas) == 0:
        return None, None

    # Grabs the frame with the largest area value ED frame
    ed = max(areas, key=lambda x: x[1])[0]

    # Grabs the frame with the smallest area value ES frame
    es = min(areas, key=lambda x: x[1])[0]

    return int(ed), int(es)
