import cv2
import numpy as np

def evaluate_output(output, min_area_ratio=0.02):
    """
    output: BGR image on black background
    min_area_ratio: minimum % of the image that must be foreground
                     or else we treat it as junk.

    Returns a score between 0 and 1.
    """

    # reject invalid
    if output is None or not isinstance(output, np.ndarray) or output.size == 0:
        return 0.0

    H, W = output.shape[:2]
    total_pixels = H * W

    # binary mask of what's being kept (non-black)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    _, bin_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    # measure connected components to find largest object kept
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    # stats[i] = [x, y, w, h, area]
    # skip label 0 (background)
    largest_area = 0
    for i in range(1, num_labels):
        area_i = stats[i, 4]
        if area_i > largest_area:
            largest_area = area_i

    area_ratio = largest_area / total_pixels  # how big is the biggest object

    # if the biggest object is super tiny (<2%), kill the score
    if area_ratio < min_area_ratio:
        return 0.0

    # sharpness of the kept region
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharp_norm = min(sharp / 500.0, 1.0)

    # encourage a reasonable size: bigger object = better, up to a point
    size_score = min(area_ratio / 0.3, 1.0)  
    # (area_ratio ~0.3 means the object covers ~30% of image, which is strong foreground)

    final_score = 0.5 * size_score + 0.5 * sharp_norm
    return float(final_score)
