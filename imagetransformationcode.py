"""
Image Similarity Proof of Concept
Compare all images in a comparison folder against a reference bank
Shows individual scores for each method per comparison
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from datetime import datetime

def load_images(path1, path2, size=(512, 512)):
    """Load and resize two images"""
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not load images. Check file paths.")
    
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)
    
    return img1, img2

def sift_similarity(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return 0.0, 0, None
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    good_matches = [m for m in matches if m.distance < 100]
    similarity = len(good_matches) / max(len(kp1), len(kp2))
    
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], 
                                   None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return similarity, len(good_matches), matched_img

def orb_similarity(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return 0.0, 0, None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    good_matches = [m for m in matches if m.distance < 50]
    similarity = len(good_matches) / max(len(kp1), len(kp2))
    
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], 
                                   None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return similarity, len(good_matches), matched_img

def ssim_similarity(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    
    return score, diff

def tensorflow_similarity(img1, img2):
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        
        model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        img1_resized = cv2.resize(img1_rgb, (224, 224))
        img2_resized = cv2.resize(img2_rgb, (224, 224))
        
        img1_batch = preprocess_input(np.expand_dims(img1_resized, axis=0))
        img2_batch = preprocess_input(np.expand_dims(img2_resized, axis=0))
        
        emb1 = model.predict(img1_batch, verbose=0)
        emb2 = model.predict(img2_batch, verbose=0)
        
        similarity = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
        
        return float(similarity)
    except ImportError:
        return None

def clip_similarity(img1, img2):
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        pil_img1 = Image.fromarray(img1_rgb)
        pil_img2 = Image.fromarray(img2_rgb)
        
        inputs1 = processor(images=pil_img1, return_tensors="pt")
        inputs2 = processor(images=pil_img2, return_tensors="pt")
        
        with torch.no_grad():
            emb1 = model.get_image_features(**inputs1)
            emb2 = model.get_image_features(**inputs2)
        
        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
        similarity = (emb1 @ emb2.T).item()
        
        return similarity
    except ImportError:
        return None

def vgg16_similarity_from_image(img1, img2):
    try:
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
        from tensorflow.keras.preprocessing.image import img_to_array

        model = VGG16(weights='imagenet', include_top=False, pooling='avg')

        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img1_resized = cv2.resize(img1_rgb, (224, 224))
        img2_resized = cv2.resize(img2_rgb, (224, 224))

        x1 = preprocess_input(np.expand_dims(img_to_array(img1_resized), axis=0))
        x2 = preprocess_input(np.expand_dims(img_to_array(img2_resized), axis=0))

        feat1 = model.predict(x1, verbose=0).flatten()
        feat2 = model.predict(x2, verbose=0).flatten()

        cos_sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        return float(cos_sim)

    except Exception as e:
        print(f"VGG16 similarity failed: {e}")
        return None

def compare_single_pair(img1, img2, method='VGG16'):
    """Compare two images using specified method"""
    if method == 'SIFT':
        score, _, _ = sift_similarity(img1, img2)
    elif method == 'ORB':
        score, _, _ = orb_similarity(img1, img2)
    elif method == 'SSIM':
        score, _ = ssim_similarity(img1, img2)
    elif method == 'TensorFlow':
        score = tensorflow_similarity(img1, img2)
    elif method == 'CLIP':
        score = clip_similarity(img1, img2)
    elif method == 'VGG16':
        score = vgg16_similarity_from_image(img1, img2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return score if score is not None else 0.0

def batch_compare_folders(comparison_folder, reference_folder, output_folder, methods=['VGG16', 'CLIP', 'SSIM']):
    """
    Compare all images in comparison_folder against all images in reference_folder
    Print individual scores for each method for every comparison
    """
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    comparison_files = [f for f in os.listdir(comparison_folder) 
                       if f.lower().endswith(valid_extensions)]
    reference_files = [f for f in os.listdir(reference_folder) 
                      if f.lower().endswith(valid_extensions)]
    
    print(f"Found {len(comparison_files)} comparison images")
    print(f"Found {len(reference_files)} reference images")
    print(f"Using methods: {', '.join(methods)}")
    
    all_results = []
    
    # Process each comparison image
    for comp_idx, comp_file in enumerate(comparison_files, 1):
        print(f"\n{'='*80}")
        print(f"COMPARISON IMAGE [{comp_idx}/{len(comparison_files)}]: {comp_file}")
        print(f"{'='*80}")
        
        comp_path = os.path.join(comparison_folder, comp_file)
        comp_img = cv2.imread(comp_path)
        comp_img = cv2.resize(comp_img, (512, 512))
        
        # Store results for this comparison image
        comp_results = []
        
        # Compare against all reference images
        for ref_idx, ref_file in enumerate(reference_files, 1):
            print(f"\nReference Image [{ref_idx}/{len(reference_files)}]: {ref_file}")
            print("-" * 80)
            
            ref_path = os.path.join(reference_folder, ref_file)
            ref_img = cv2.imread(ref_path)
            ref_img = cv2.resize(ref_img, (512, 512))
            
            result = {
                'comparison_image': comp_file,
                'reference_image': ref_file
            }
            
            # Run each method and print individual scores
            for method in methods:
                score = compare_single_pair(comp_img, ref_img, method=method)
                result[f'{method}_score'] = score
                print(f"  {method:15s}: {score:.6f}")
            
            comp_results.append(result)
        
        # Show top 3 for each method
        print(f"\n{'='*80}")
        print(f"TOP 3 MATCHES FOR: {comp_file}")
        print(f"{'='*80}")
        for method in methods:
            sorted_results = sorted(comp_results, key=lambda x: x[f'{method}_score'], reverse=True)
            top3 = sorted_results[:3]
            
            print(f"\n{method} - Top 3:")
            for i, res in enumerate(top3, 1):
                print(f"  {i}. {res['reference_image']:40s} Score: {res[f'{method}_score']:.6f}")
            
            # Create visualization for top 3
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Show comparison image
            axes[0].imshow(cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f'Comparison:\n{comp_file}', fontsize=10)
            axes[0].axis('off')
            
            # Show top 3 matches
            for i, res in enumerate(top3):
                ref_path = os.path.join(reference_folder, res['reference_image'])
                ref_img_display = cv2.imread(ref_path)
                ref_img_display = cv2.resize(ref_img_display, (512, 512))
                
                axes[i+1].imshow(cv2.cvtColor(ref_img_display, cv2.COLOR_BGR2RGB))
                axes[i+1].set_title(f'Match #{i+1}\n{res["reference_image"]}\n{method}: {res[f"{method}_score"]:.4f}', 
                                   fontsize=10)
                axes[i+1].axis('off')
            
            plt.tight_layout()
            output_filename = f"{os.path.splitext(comp_file)[0]}_{method}_top3.png"
            output_path = os.path.join(output_folder, output_filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        # Feature matching visualizations for top match
        best_overall = sorted(comp_results, key=lambda x: x[f'{methods[0]}_score'], reverse=True)[0]
        best_ref_path = os.path.join(reference_folder, best_overall['reference_image'])
        best_ref_img = cv2.imread(best_ref_path)
        best_ref_img = cv2.resize(best_ref_img, (512, 512))
        
        # SIFT and ORB feature matching visualizations
        print(f"\nGenerating feature matching visualizations for best match: {best_overall['reference_image']}")
        
        # SIFT 
        sift_score, sift_matches, sift_img = sift_similarity(comp_img, best_ref_img)
        if sift_img is not None:
            sift_output = os.path.join(output_folder, f"{os.path.splitext(comp_file)[0]}_SIFT_matches.png")
            cv2.imwrite(sift_output, sift_img)
            print(f"  Saved SIFT matching: {os.path.basename(sift_output)} (score: {sift_score:.4f}, {sift_matches} matches)")
        
        # ORB 
        orb_score, orb_matches, orb_img = orb_similarity(comp_img, best_ref_img)
        if orb_img is not None:
            orb_output = os.path.join(output_folder, f"{os.path.splitext(comp_file)[0]}_ORB_matches.png")
            cv2.imwrite(orb_output, orb_img)
            print(f"  Saved ORB matching: {os.path.basename(orb_output)} (score: {orb_score:.4f}, {orb_matches} matches)")
        
        # Add all results to master list
        all_results.extend(comp_results)
    
    # Save as CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_folder, 'all_similarity_scores.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*80}")
    print(f"Saved all scores to: {csv_path}")
    
    # Save summary JSON
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'comparison_folder': comparison_folder,
        'reference_folder': reference_folder,
        'num_comparison_images': len(comparison_files),
        'num_reference_images': len(reference_files),
        'methods_used': methods,
        'total_comparisons': len(all_results)
    }
    
    json_path = os.path.join(output_folder, 'summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {json_path}")
    
    # Create summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    for method in methods:
        scores = df[f'{method}_score']
        print(f"\n{method}:")
        print(f"  Mean score: {scores.mean():.4f}")
        print(f"  Max score: {scores.max():.4f}")
        print(f"  Min score: {scores.min():.4f}")
    
    return df

if __name__ == "__main__":
    # PATHS
    comparison_folder = "C:/Users/huichris/Downloads/original"
    reference_folder = "C:/Users/huichris/Downloads/testimages"
    output_folder = "C:/Users/huichris/Downloads/similarityfactortest1"
    
    # Choose methods
    methods = ['SIFT', 'SSIM', 'CLIP', 'ORB','VGG16', 'TensorFlow']
    
    try:
        results_df = batch_compare_folders(
            comparison_folder=comparison_folder,
            reference_folder=reference_folder,
            output_folder=output_folder,
            methods=methods
        )
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE!")
        print("="*80)
        print(f"Check the output folder for all visualizations and data files")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()