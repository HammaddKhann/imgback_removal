import cv2
import numpy as np

def evaluate_output(output, min_area_ratio=0.02, debug=False):
    
    if output is None or not isinstance(output, np.ndarray) or output.size == 0:
        return 0.0

    H, W = output.shape[:2]
    total_pixels = H * W

    # Create binary mask of foreground
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    _, bin_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    # Get connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8
    )
    
    if num_labels <= 1:
        return 0.0
    
    # Find largest component (main object)
    largest_label = 1 + np.argmax(stats[1:, 4])
    largest_area = stats[largest_label, 4]
    area_ratio = largest_area / total_pixels

    # ============================================
    # 1. SIZE SCORE (25%)
    # ============================================
    if area_ratio < min_area_ratio:
        return 0.0
    
    # Optimal range: 15-60% of image
    if area_ratio < 0.15:
        size_score = area_ratio / 0.15
    elif area_ratio <= 0.60:
        size_score = 1.0
    else:
        # Penalize if object fills entire image (likely poor removal)
        size_score = max(0.3, 1.0 - (area_ratio - 0.60) / 0.40)

    # ============================================
    # 2. EDGE QUALITY SCORE (25%)
    # ============================================
    largest_mask = (labels == largest_label).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(
        largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        edge_score = 0.0
    else:
        largest_contour = max(contours, key=cv2.contourArea)
        
        perimeter = cv2.arcLength(largest_contour, True)
        area = cv2.contourArea(largest_contour)
        
        if area > 0 and perimeter > 0:
            # Isoperimetric quotient (measures compactness/smoothness)
            # Perfect circle = 1.0, jagged/complex shapes < 1.0
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # We don't expect perfect circles, but jagged edges will score low
            # Scale up slightly since most objects aren't circular
            edge_score = min(1.0, circularity * 2.5)
        else:
            edge_score = 0.0

    # ============================================
    # 3. COMPLETENESS SCORE (20%)
    # ============================================
    # Check if background is truly black (properly removed)
    background_mask = cv2.bitwise_not(largest_mask)
    background_pixels = cv2.bitwise_and(gray, gray, mask=background_mask)
    
    background_nonzero = cv2.countNonZero(background_pixels)
    background_total = cv2.countNonZero(background_mask)
    
    if background_total > 0:
        # Ratio of black pixels in background
        background_removed_ratio = 1.0 - (background_nonzero / background_total)
        completeness_score = max(0.0, background_removed_ratio)
    else:
        completeness_score = 1.0

    # ============================================
    # 4. PRESERVATION SCORE (15%)
    # ============================================
    # Check for holes/gaps in foreground
    foreground_pixels = cv2.bitwise_and(gray, gray, mask=largest_mask)
    foreground_total = cv2.countNonZero(largest_mask)
    foreground_filled = cv2.countNonZero(foreground_pixels)
    
    if foreground_total > 0:
        # How much of the foreground mask is actually filled with content
        fill_ratio = foreground_filled / foreground_total
        preservation_score = fill_ratio
    else:
        preservation_score = 0.0

    # ============================================
    # 5. SHARPNESS SCORE (15%)
    # ============================================
    # Use Laplacian variance on foreground only
    foreground_only = cv2.bitwise_and(gray, gray, mask=largest_mask)
    
    if cv2.countNonZero(largest_mask) > 0:
        laplacian_var = cv2.Laplacian(foreground_only, cv2.CV_64F).var()
        # Normalize: typical good values range from 100-500
        # Higher = sharper, but too high can mean noise
        sharpness_score = min(laplacian_var / 400.0, 1.0)
    else:
        sharpness_score = 0.0

    # ============================================
    # FINAL WEIGHTED SCORE
    # ============================================
    final_score = (
        0.25 * size_score +
        0.20 * edge_score +
        0.25 * completeness_score +
        0.15 * preservation_score +
        0.15 * sharpness_score
    )

    # Debug output
    if debug:
        print(f"  Size: {size_score:.3f} | Edge: {edge_score:.3f} | "
              f"Complete: {completeness_score:.3f} | Preserve: {preservation_score:.3f} | "
              f"Sharp: {sharpness_score:.3f} → FINAL: {final_score:.3f}")

    return float(final_score)


def evaluate_batch(outputs, debug=False):
    """
    Evaluate multiple outputs and return scores.
    
    Args:
        outputs: dict of {model_name: output_image}
        debug: if True, prints detailed scores
    
    Returns:
        dict: {model_name: score}
    """
    scores = {}
    
    if debug:
        print("\n" + "="*60)
        print("DETAILED EVALUATION SCORES")
        print("="*60)
    
    for model_name, output in outputs.items():
        if output is None:
            scores[model_name] = 0.0
            continue
        
        if debug:
            print(f"\n{model_name}:")
        
        score = evaluate_output(output, debug=debug)
        scores[model_name] = score
    
    if debug:
        print("\n" + "="*60)
        print("FINAL RANKINGS")
        print("="*60)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(sorted_scores, 1):
            marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"{marker} {i}. {name}: {score:.4f}")
        print("="*60 + "\n")
    
    return scores


def compare_outputs(outputs, original=None, save_comparison=False, output_path="comparison.jpg"):
    """
    Create a visual comparison of all outputs (optional utility function).
    
    Args:
        outputs: dict of {model_name: output_image}
        original: original input image (optional)
        save_comparison: if True, saves comparison image
        output_path: path to save comparison
    """
    import matplotlib.pyplot as plt
    
    num_outputs = len(outputs)
    if original is not None:
        num_outputs += 1
    
    fig, axes = plt.subplots(1, num_outputs, figsize=(5*num_outputs, 5))
    if num_outputs == 1:
        axes = [axes]
    
    idx = 0
    
    if original is not None:
        axes[idx].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[idx].set_title("Original")
        axes[idx].axis('off')
        idx += 1
    
    scores = evaluate_batch(outputs, debug=False)
    
    for model_name, output in outputs.items():
        if output is not None:
            axes[idx].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            score = scores[model_name]
            axes[idx].set_title(f"{model_name}\nScore: {score:.4f}")
        else:
            axes[idx].text(0.5, 0.5, 'Failed', ha='center', va='center')
            axes[idx].set_title(f"{model_name}\nFailed")
        axes[idx].axis('off')
        idx += 1
    
    plt.tight_layout()
    
    if save_comparison:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to {output_path}")
    
    plt.show()