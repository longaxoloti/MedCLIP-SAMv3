import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
import os
import argparse
from sklearn.cluster import KMeans
from tqdm import tqdm
from laplacian_refinement import laplacian_guided_refine

np.random.seed(10)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def postprocess_crf(args):
    files = os.listdir(args.sal_path)

    os.makedirs(args.output_path, exist_ok=True)

    for file in tqdm(files):

        img = cv2.imread(args.input_path+'/'+file, 1)
        annos = cv2.imread(args.sal_path+'/'+file, 0)
        annos = cv2.resize(annos, (img.shape[1], img.shape[0]))
        output = args.output_path+'/'+file

        # Setup the CRF model
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], args.m)

        anno_norm = annos / 255.
        n_energy = -np.log((1.0 - anno_norm + args.epsilon)) / (args.tau * sigmoid(1 - anno_norm))
        p_energy = -np.log(anno_norm + args.epsilon) / (args.tau * sigmoid(anno_norm))

        U = np.zeros((args.m, img.shape[0] * img.shape[1]), dtype='float32')
        U[0, :] = n_energy.flatten()
        U[1, :] = p_energy.flatten()

        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=args.gaussian_sxy, compat=3)
        d.addPairwiseBilateral(sxy=args.bilateral_sxy, srgb=args.bilateral_srgb, rgbim=img, compat=5)

        # Do the inference
        Q = d.inference(1)
        map = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))

        # Save the output as image
        segmented_image = map.astype('uint8')*255

        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(segmented_image)
        sizes = stats[:, cv2.CC_STAT_AREA]
        
        # Sort sizes (ignoring the background at index 0)
        sorted_sizes = sorted(sizes[1:], reverse=True) 
        
        # Determine the top K sizes
        top_k_sizes = sorted_sizes[:args.num_contours]
        
        im_result = np.zeros_like(im_with_separated_blobs)
        
        for index_blob in range(1, nb_blobs):
            if sizes[index_blob] in top_k_sizes:
                im_result[im_with_separated_blobs == index_blob] = 255
        
        segmented_image = im_result

        cv2.imwrite(output, segmented_image)
        
def postprocess_thresholding(args):
    files = os.listdir(args.sal_path)

    os.makedirs(args.output_path, exist_ok=True)

    for file in tqdm(files):

        annos = cv2.imread(args.sal_path+'/'+file, 0)
        output = args.output_path+'/'+file

        annos = annos / 255.
        annos = (annos > args.threshold).astype(np.uint8)
        segmented_image = annos * 255

        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(segmented_image)
        sizes = stats[:, cv2.CC_STAT_AREA]
        
        # Sort sizes (ignoring the background at index 0)
        sorted_sizes = sorted(sizes[1:], reverse=True) 
        
        # Determine the top K sizes
        top_k_sizes = sorted_sizes[:args.num_contours]
        
        im_result = np.zeros_like(im_with_separated_blobs)
        
        for index_blob in range(1, nb_blobs):
            if sizes[index_blob] in top_k_sizes:
                im_result[im_with_separated_blobs == index_blob] = 255
        
        segmented_image = im_result

        cv2.imwrite(output, segmented_image)

def postprocess_kmeans(args):
    
    files = os.listdir(args.sal_path)

    os.makedirs(args.output_path, exist_ok=True)

    for file in tqdm(files):

        attn_weights = cv2.imread(args.sal_path+'/'+file, 0) / 255.0
        h, w = attn_weights.shape
        
        # ====== STEP 1: LOAD AND REFINE SALIENCY MAP ======
        if args.use_lg_sr:
            original_image = cv2.imread(os.path.join(args.input_path, file), 1)
            
            if original_image is not None:
                try:
                    # Convert saliency map to 0-255 range for LG-SR processing
                    attn_weights_uint8 = (attn_weights * 255).astype(np.uint8)
                    
                    # Apply Laplacian-Guided Saliency Refinement on original saliency map
                    refined_saliency = laplacian_guided_refine(
                        attn_weights_uint8,
                        original_image,
                        alpha=args.lg_sr_alpha,
                        edge_threshold=args.lg_sr_edge_threshold,
                        use_watershed=args.lg_sr_use_watershed
                    )
                    
                    # Convert back to 0-1 range
                    attn_weights = refined_saliency.astype(np.float32) / 255.0
                    
                    if args.verbose:
                        print(f"  {file}: LG-SR applied to saliency map")
                        
                except Exception as e:
                    if args.verbose:
                        print(f"  Warning: LG-SR failed for {file}: {str(e)}")
                    # Continue with original attn_weights
        # ================================================
        
        # STEP 2: THRESHOLDING - OTSU or K-MEANS
        if args.use_otsu:
            # Use Otsu automatic thresholding
            attn_weights_uint8 = (attn_weights * 255).astype(np.uint8)
            threshold_value, segmented_image = cv2.threshold(
                attn_weights_uint8, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            if args.verbose:
                print(f"  {file}: Otsu threshold = {threshold_value}")
        else:
            # Use K-means clustering (original method)
            kmeans = KMeans(n_clusters=2, random_state=10)
            image = cv2.resize(attn_weights, (256, 256), interpolation=cv2.INTER_NEAREST)
            flat_image = image.reshape(-1, 1)

            labels = kmeans.fit_predict(flat_image)

            segmented_image = labels.reshape(256, 256)

            centroids = kmeans.cluster_centers_.flatten()

            # Identify the background cluster (assuming it has the lowest centroid value)
            background_cluster = np.argmin(centroids)

            # Mark background pixels as 0 and foreground pixels as 1
            segmented_image = np.where(segmented_image == background_cluster, 0, 1)

            segmented_image = cv2.resize(segmented_image, (w, h), interpolation=cv2.INTER_NEAREST)
            segmented_image = segmented_image.astype(np.uint8) * 255

        # STEP 3: MORPHOLOGICAL REFINEMENT (Optional)
        if args.use_morph_refine:
            # Define kernel size based on image size
            kernel_size = max(3, min(7, int(min(h, w) * 0.01)))  # Adaptive kernel size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Erosion: Remove small noise blobs and thin boundaries
            eroded = cv2.erode(segmented_image, kernel, iterations=args.morph_erode_iters)
            
            # Dilation: Recover the main region size
            segmented_image = cv2.dilate(eroded, kernel, iterations=args.morph_dilate_iters)
            
            if args.verbose:
                print(f"  {file}: Morphological refinement (kernel={kernel_size}, erode={args.morph_erode_iters}, dilate={args.morph_dilate_iters})")

        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(segmented_image)
        sizes = stats[:, cv2.CC_STAT_AREA]
        
        # Sort sizes (ignoring the background at index 0)
        sorted_sizes = sorted(sizes[1:], reverse=True) 
        
        # Determine the top K sizes
        top_k_sizes = sorted_sizes[:args.num_contours]
        
        im_result = np.zeros_like(im_with_separated_blobs)
        
        for index_blob in range(1, nb_blobs):
            if sizes[index_blob] in top_k_sizes:
                im_result[im_with_separated_blobs == index_blob] = 255
        
        segmented_image = im_result

        cv2.imwrite(args.output_path+'/'+file, segmented_image)


def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gaussian-sxy', type=int, default=5,
                        help="Gaussian sxy value for CRF")
    parser.add_argument('--bilateral-sxy', type=int, default=5,
                        help="Bilateral sxy value for CRF")
    parser.add_argument('--bilateral-srgb', type=int, default=3,
                        help="Bilateral srgb value for CRF")
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help="Epsilon value for CRF")
    parser.add_argument('--m', type=int, default=2,
                        help="Number of classes in the saliency map")
    parser.add_argument('--tau', type=float, default=1.05,
                        help="Tau value for CRF")
    parser.add_argument('--threshold', type=float, default=0.3,
                        help="Threshold value for thresholding")
    parser.add_argument('--input-path', type=str, default='images',
                        help="Path to the images")
    parser.add_argument('--sal-path', type=str, default='cams',
                        help="Path to the saliency maps")
    parser.add_argument('--output-path', type=str, default='output',
                        help="Output path of CRF postprocessed samples")
    parser.add_argument('--postprocess', type=str, default='kmeans', choices=['crf', 'thresholding', 'kmeans'],
                        help="Postprocessing method to use (crf/thresholding/kmeans)")
    parser.add_argument('--filter', action='store_true',
                        help="Whether to filter small clusters")
    parser.add_argument('--min-size', type=int, default=100,
                        help="Minimum size of clusters to keep")
    parser.add_argument('--num-contours', type=int, default=1, help="Number of contours to keep")
    
    # ====== LG-SR (Laplacian-Guided Saliency Refinement) Arguments ======
    parser.add_argument('--use-lg-sr', action='store_true',
                        help="Enable Laplacian-Guided Saliency Refinement (default: disabled)")
    parser.add_argument('--lg-sr-alpha', type=float, default=0.5,
                        help="Fusion gain factor for LG-SR. Higher = more edge reinforcement (default 0.5)")
    parser.add_argument('--lg-sr-edge-threshold', type=float, default=0.1,
                        help="Threshold for determining weak edges in LG-SR. "
                             "If mean edge strength < threshold, use fallback (default 0.1)")
    parser.add_argument('--lg-sr-use-watershed', action='store_true', default=False,
                        help="Use watershed fallback instead of morphological (default: False, use morphological)")
    parser.add_argument('--verbose', action='store_true',
                        help="Print verbose debug information")
    # ====================================================================
    
    # ====== Thresholding Options ======
    parser.add_argument('--use-otsu', action='store_true',
                        help="Use Otsu automatic thresholding instead of K-means (default: False, use K-means)")
    # ===================================
    
    # ====== Morphological Refinement Options ======
    parser.add_argument('--use-morph-refine', action='store_true',
                        help="Apply morphological refinement (erosion + dilation) after thresholding (default: False)")
    parser.add_argument('--morph-erode-iters', type=int, default=1,
                        help="Number of erosion iterations to remove noise (default: 1)")
    parser.add_argument('--morph-dilate-iters', type=int, default=1,
                        help="Number of dilation iterations to recover size (default: 1)")
    # ==============================================

    return parser.parse_args()
if __name__ == '__main__':
    args = get_parser()
    print("Postprocessing started...")
    if(args.postprocess == 'crf'):
        postprocess_crf(args)
    elif(args.postprocess == 'thresholding'):
        postprocess_thresholding(args)
    elif(args.postprocess == 'kmeans'):
        postprocess_kmeans(args)
    print("Postprocessing done!")

