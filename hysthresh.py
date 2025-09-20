import numpy as np
from skimage.measure import label
from skimage.morphology import remove_small_objects

def hysthresh(im, T1, T2):
    """
    Performs hysteresis thresholding of an image.
    All pixels with values above threshold T1 are marked as edges.
    All pixels that are connected to points that have been marked as edges
    and with values above threshold T2 are also marked as edges. Eight
    connectivity is used.
    Args:
        im: 2D numpy array (image)
        T1: upper threshold value
        T2: lower threshold value
    Returns:
        bw: thresholded image (binary mask)
    """
    # Ensure T1 is the upper threshold
    if T1 < T2:
        T1, T2 = T2, T1

    aboveT2 = im > T2
    aboveT1 = im > T1

    # Label connected regions in aboveT2
    labeled, num = label(aboveT2, connectivity=2, return_num=True)

    # Find labels that contain at least one pixel above T1
    mask = np.zeros_like(aboveT2, dtype=bool)
    for region_label in range(1, num + 1):
        region = (labeled == region_label)
        if np.any(aboveT1 & region):
            mask |= region

    return mask.astype(np.uint8)
