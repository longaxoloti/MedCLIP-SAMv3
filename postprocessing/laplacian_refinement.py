"""
Laplacian-Guided Saliency Refinement (LG-SR) Module
====================================================
Post-processing pipeline to refine binary masks from K-means clustering
using high-frequency information from Laplacian edge detection.

Features:
- Training-free post-processing (no model fine-tuning required)
- Dual-stream fusion: semantic (binary mask) + structural (Laplacian edges)
- Adaptive fallback method for homogeneous tumors with weak edges
"""

import numpy as np
import cv2
from scipy import ndimage
import warnings


def compute_laplacian_edges(image, sigma=1.0):
    """
    Compute high-frequency edges using Laplacian pyramid.
    
    Args:
        image: Input image (uint8 or float32) - can be grayscale or RGB
        sigma: Gaussian blur standard deviation (default: 1.0)
    
    Returns:
        M_norm: Normalized edge map [0, 1] containing high-frequency information
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Ensure float type for processing
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
    
    # Step 1: Low-pass filtering (Gaussian blur)
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    img_low = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Step 2: High-pass filtering (Laplacian)
    L_hf = image - img_low
    
    # Step 3: Compute magnitude (handles both positive and negative values)
    M_freq = np.abs(L_hf)
    
    # Step 4: Normalize to [0, 1]
    M_min = M_freq.min()
    M_max = M_freq.max()
    
    if M_max > M_min:
        M_norm = (M_freq - M_min) / (M_max - M_min)
    else:
        M_norm = np.zeros_like(M_freq)
    
    return M_norm


def adaptive_fusion(binary_mask, edge_map, alpha=0.5):
    """
    Fuse semantic information (binary mask) with structural information (edges).
    
    Fusion equation:
    S_refined = S_raw * (1 + alpha * M_norm)
    
    - Multiplication acts as AND logic gate: only pixels with high semantic 
      probability AND strong edge information contribute
    
    Args:
        binary_mask: Binary mask from K-means [0, 1] float or [0, 255] uint8
        edge_map: Normalized edge map from Laplacian [0, 1]
        alpha: Gain factor for edge reinforcement (default: 0.5)
    
    Returns:
        S_refined: Refined saliency map [0, 1]
    """
    # Ensure float type
    if binary_mask.dtype == np.uint8:
        S_raw = binary_mask.astype(np.float32) / 255.0
    else:
        S_raw = binary_mask.astype(np.float32)
        
    edge_map = edge_map.astype(np.float32)
    
    # Fusion equation: S_refined = S_raw * (1 + alpha * M_norm)
    refinement_factor = 1.0 + alpha * edge_map
    S_refined = S_raw * refinement_factor
    
    # Clip to valid range [0, 1]
    S_refined = np.clip(S_refined, 0.0, 1.0)
    
    return S_refined


def detect_edge_strength(edge_map, threshold=0.1):
    """
    Assess overall edge strength in the image.
    Used to trigger fallback methods when edges are too weak.
    
    Args:
        edge_map: Normalized edge map [0, 1]
        threshold: Strength threshold (default: 0.1)
    
    Returns:
        tuple: (mean_edge_strength, is_weak_edge_bool)
    """
    # Calculate mean edge strength
    edge_strength = np.mean(edge_map)
    
    # If mean strength < threshold, consider it weak
    is_weak = edge_strength < threshold
    
    return edge_strength, is_weak


def extract_contours_from_refined(refined_saliency):
    """
    Extract binary mask from refined saliency using Otsu thresholding.
    
    Args:
        refined_saliency: Refined saliency map [0, 1]
    
    Returns:
        binary_mask: Binary mask (uint8: 0-255)
    """
    # Convert to uint8 for OpenCV operations
    saliency_uint8 = (refined_saliency * 255).astype(np.uint8)
    
    # Otsu's thresholding for automatic threshold selection
    try:
        otsu_thresh, binary = cv2.threshold(
            saliency_uint8, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    except Exception as e:
        warnings.warn(f"Otsu thresholding failed: {str(e)}. Using 50% threshold.")
        binary = (refined_saliency > 0.5).astype(np.uint8) * 255
    
    return binary


def morphological_fallback(binary_mask, iterations=2, method='close'):
    """
    Fallback method for homogeneous tumors with weak edge detection.
    Uses morphological operations to refine boundaries.
    
    Args:
        binary_mask: Input binary mask (uint8: 0-255 or float 0-1)
        iterations: Number of morphological iterations (default: 2)
        method: 'close' (dilation+erosion) or 'open' (erosion+dilation)
    
    Returns:
        refined_mask: Refined binary mask with smoother boundaries (uint8)
    """
    # Ensure uint8 for morphological ops
    if binary_mask.dtype == np.float32 or binary_mask.dtype == np.float64:
        mask_uint8 = (binary_mask * 255).astype(np.uint8)
    else:
        mask_uint8 = binary_mask.astype(np.uint8)
    
    # Create kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Apply morphological closing (dilation -> erosion)
    # Removes small holes, smooths boundaries, connects nearby regions
    if method == 'close':
        refined_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, 
                                        iterations=iterations)
    # Or opening (erosion -> dilation)
    # Removes small objects, separates touching regions
    elif method == 'open':
        refined_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel,
                                        iterations=iterations)
    else:
        refined_mask = mask_uint8
    
    return refined_mask


def watershed_fallback(image, binary_mask, kernel_size=15):
    """
    Fallback method using Watershed algorithm for better boundary definition.
    Works well for partially separated regions or blurry boundaries.
    
    Args:
        image: Original grayscale or RGB image
        binary_mask: Binary mask from K-means (uint8)
        kernel_size: Morphological kernel size (default: 15)
    
    Returns:
        watershed_mask: Refined mask using watershed (uint8)
    """
    # Ensure proper types
    if image.dtype == np.float32 or image.dtype == np.float64:
        image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)
    
    if binary_mask.dtype != np.uint8:
        binary_mask = (binary_mask * 255).astype(np.uint8)
    
    # Ensure grayscale
    if len(image_uint8.shape) == 3:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    
    try:
        # Step 1: Morphological opening (noise removal)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        sure_bg = cv2.dilate(binary_mask, kernel, iterations=3)
        
        # Step 2: Find sure foreground
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Step 3: Find unknown region (boundary)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Step 4: Label markers
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # Foreground markers > 1, background = 1
        markers[unknown == 255] = 0  # Unknown region = 0
        
        # Step 5: Apply watershed on color image
        image_color = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(image_color, markers)
        
        # Convert markers to binary mask (foreground regions > 1)
        watershed_mask = np.zeros_like(markers, dtype=np.uint8)
        watershed_mask[markers > 1] = 255
        
    except Exception as e:
        warnings.warn(f"Watershed failed: {str(e)}. Using morphological fallback.")
        watershed_mask = morphological_fallback(binary_mask)
    
    return watershed_mask


def laplacian_guided_refine(binary_mask, original_image, alpha=0.5, 
                           edge_threshold=0.1, use_watershed=False):
    """
    Main function: Laplacian-Guided Saliency Refinement (LG-SR)
    
    Refines binary segmentation mask using high-frequency edge information
    with intelligent fallback for weak edges.
    
    Pipeline:
    1. Extract high-frequency edges via Laplacian pyramid
    2. Assess edge strength; trigger fallback if weak
    3. Fuse semantic (mask) + structural (edges) information
    4. Produce refined binary mask
    
    Args:
        binary_mask: Input binary mask from K-means (uint8: 0-255 or float 0-1)
        original_image: Original image (numpy array, grayscale or RGB)
        alpha: Fusion gain factor (default: 0.5)
        edge_threshold: Edge strength threshold for fallback (default: 0.1)
        use_watershed: Use watershed fallback instead of morphological (default: False)
    
    Returns:
        refined_mask: Refined binary mask (uint8: 0-255)
    """
    # Validate inputs
    if binary_mask is None or original_image is None:
        warnings.warn("Invalid inputs. Returning original mask.")
        if binary_mask is not None:
            return (binary_mask * 255).astype(np.uint8) if binary_mask.max() <= 1 else binary_mask.astype(np.uint8)
        else:
            return None
    
    try:
        # Convert binary_mask to float [0, 1] for processing
        if binary_mask.dtype == np.uint8:
            mask_float = binary_mask.astype(np.float32) / 255.0
        else:
            mask_float = binary_mask.astype(np.float32)
        
        # Step 1: Compute Laplacian edges
        edge_map = compute_laplacian_edges(original_image, sigma=1.0)
        
        # Step 2: Assess edge strength
        edge_strength, is_weak_edge = detect_edge_strength(edge_map, threshold=edge_threshold)
        
        # Step 3: Apply appropriate refinement strategy
        if is_weak_edge:
            # Edge detection too weak: use fallback methods
            warnings.warn(f"Weak edge detection (strength={edge_strength:.4f}). "
                         f"Using {'watershed' if use_watershed else 'morphological'} fallback.")
            
            if use_watershed:
                refined_uint8 = watershed_fallback(original_image, 
                                                   (mask_float * 255).astype(np.uint8))
            else:
                refined_uint8 = morphological_fallback(mask_float, iterations=2, method='close')
        else:
            # Edges are strong: apply Laplacian-guided fusion
            S_refined = adaptive_fusion(mask_float, edge_map, alpha=alpha)
            
            # Convert refined saliency to binary via thresholding
            refined_uint8 = extract_contours_from_refined(S_refined)
        
        # Step 4: Final light morphological cleanup for smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined_mask = cv2.morphologyEx(refined_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return refined_mask
        
    except Exception as e:
        warnings.warn(f"LG-SR refinement failed: {str(e)}. Returning original mask.")
        if binary_mask.dtype == np.uint8:
            return binary_mask
        else:
            return (binary_mask * 255).astype(np.uint8)
