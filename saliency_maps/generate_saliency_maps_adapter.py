import warnings
warnings.filterwarnings('ignore')

import os
import sys
import builtins
from pathlib import Path
import torch
import cv2
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import argparse
import pandas as pd
import itertools
import json
import random
from PIL import Image

from scripts.methods import vision_heatmap_iba
from text_prompts import *

# Disable parallel tokenization warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root for adapter imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from biomedclip_finetuning.frequency_adapter import inject_frequency_adapters


def _patch_clip_symbols_for_biomedclip():
    """Patch missing CLIP symbols for BiomedCLIP HF remote modules.
    
    BiomedCLIP's remote module references CLIP classes that may not be in global scope.
    We inject them into builtins so they can be found during module loading.
    """
    try:
        from transformers.models.clip.modeling_clip import (
            CLIPMLP, CLIPVisionEmbeddings, CLIPVisionTransformer,
            CLIPTextEmbeddings, CLIPEncoder, CLIPAttention,
            CLIPTextTransformer, CLIPOutput, CLIPPreTrainedModel
        )
        from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
        
        builtins.CLIPMLP = CLIPMLP
        builtins.CLIPVisionEmbeddings = CLIPVisionEmbeddings
        builtins.CLIPVisionTransformer = CLIPVisionTransformer
        builtins.CLIPTextEmbeddings = CLIPTextEmbeddings
        builtins.CLIPEncoder = CLIPEncoder
        builtins.CLIPAttention = CLIPAttention
        builtins.CLIPTextTransformer = CLIPTextTransformer
        builtins.CLIPOutput = CLIPOutput
        builtins.CLIPPreTrainedModel = CLIPPreTrainedModel
        builtins.BaseModelOutput = BaseModelOutput
        builtins.BaseModelOutputWithPooling = BaseModelOutputWithPooling
    except Exception as e:
        print(f"[WARN] Could not patch all CLIP symbols: {e}")
    except ImportError as e:
        print(f"[WARN] Could not import CLIP symbols (will try at runtime): {e}")


def calculate_dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    dice_coefficient = (2.0 * intersection) / (mask1.sum() + mask2.sum())
    return dice_coefficient


def evaluate_on_sample(model, processor, tokenizer, text, image_paths, args):
    dice_scores = []
    for image_id in tqdm(image_paths):
        try:
            image = Image.open(f"{args.val_path}/{image_id}").convert('RGB')
        except Exception:
            continue

        image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(args.device)
        text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(args.device)

        vmap = vision_heatmap_iba(
            text_ids,
            image_feat,
            model,
            args.vlayer,
            args.vbeta,
            args.vvar,
            ensemble=args.ensemble,
            progbar=False
        )

        gt_path = args.val_path.replace("images", "masks")
        gt_mask = np.array(Image.open(f"{gt_path}/{image_id}").convert("L"))

        vmap_resized = cv2.resize(np.array(vmap), (gt_mask.shape[1], gt_mask.shape[0]))
        cam_img = vmap_resized > 0.3

        dice_score = calculate_dice_coefficient(gt_mask.astype(bool), cam_img.astype(bool))
        dice_scores.append(dice_score)

    average_dice = np.mean(dice_scores)
    return average_dice


def hyper_opt(model, processor, tokenizer, text, args):
    print("Running Hyperparameter Optimization ...")

    vbeta_list = [0.1, 1.0, 2.0]
    vvar_list = [0.1, 1.0, 2.0]
    layers_list = [7, 8, 9]

    hyperparameter_combinations = list(itertools.product(vbeta_list, vvar_list, layers_list))
    all_image_ids = sorted(os.listdir(args.val_path))

    results = []
    for combo in hyperparameter_combinations:
        vbeta, vvar, layer = combo
        args.vbeta = vbeta
        args.vvar = vvar
        args.vlayer = layer

        sample_dice_scores = []
        print(f"Evaluating combination: vbeta={vbeta}, vvar={vvar}, layer={layer}")

        for i in range(3):
            random.seed(i)
            sampled_images = random.sample(all_image_ids, 1)
            avg_dice = evaluate_on_sample(model, processor, tokenizer, text, sampled_images, args)
            sample_dice_scores.append(avg_dice)
            print(f"  Sample {i+1}: Average Dice Score = {avg_dice}")

        mean_dice = np.mean(sample_dice_scores)
        results.append({
            'vbeta': vbeta,
            'vvar': vvar,
            'vlayer': layer,
            'average_dice': mean_dice
        })
        print(f"Mean Dice Score for this combination: {mean_dice}\n")

    results_df = pd.DataFrame(results)
    best_combo = results_df.loc[results_df['average_dice'].idxmax()]

    print("Best Hyperparameter Combination:")
    print(best_combo)
    print("\n")

    return best_combo


def load_biomedclip_with_adapters(args):
    _patch_clip_symbols_for_biomedclip()
    base_checkpoint_file = None
    if args.checkpoint_path:
        if os.path.isfile(args.checkpoint_path):
            base_checkpoint_file = args.checkpoint_path
            model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
        else:
            model = AutoModel.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)

    processor = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)

    model = inject_frequency_adapters(
        model,
        adapter_layers=args.adapter_layers,
        rank_k=args.rank_k,
        gamma_init=args.gamma_init,
        adapter_dropout=args.adapter_dropout,
        verbose=True
    )

    if base_checkpoint_file:
        checkpoint = torch.load(base_checkpoint_file, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading base checkpoint file: {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading base checkpoint file: {len(unexpected)}")

    if args.adapter_checkpoint:
        checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading adapter checkpoint: {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading adapter checkpoint: {len(unexpected)}")

    model = model.to(args.device)
    model.eval()

    return model, processor, tokenizer


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("Loading BiomedCLIP with frequency adapters ...")
    model, processor, tokenizer = load_biomedclip_with_adapters(args)

    text = None
    if args.text_prompt:
        text = args.text_prompt
        print(f"Using provided text prompt: {text}")
    elif not args.reproduce and not args.json_path:
        text = str(input("Enter the text: "))

    if args.hyper_opt:
        if text is None:
            raise ValueError("Cannot run hyperparameter optimization without a fixed text prompt")
        best_combo = hyper_opt(model, processor, tokenizer, text, args)
        args.vbeta = best_combo['vbeta']
        args.vvar = best_combo['vvar']
        args.vlayer = int(best_combo['vlayer'])

    print("Generating Saliency Maps ...")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    json_decoded = None
    if args.reproduce or args.json_path:
        if not args.json_path:
            raise ValueError("--json-path is required when using --reproduce")
        with open(args.json_path) as json_file:
            json_decoded = json.load(json_file)

    for image_id in tqdm(sorted(os.listdir(args.input_path))):
        if image_id in os.listdir(args.output_path):
            continue
        try:
            image = Image.open(f"{args.input_path}/{image_id}").convert('RGB')
        except Exception:
            print(f"Unable to load image at {image_id}", flush=True)
            continue

        if json_decoded is not None:
            text = json_decoded.get(image_id, text)
            if text is None:
                print(f"Missing prompt for {image_id}, skipping", flush=True)
                continue

        image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(args.device)
        text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(args.device)

        vmap = vision_heatmap_iba(
            text_ids,
            image_feat,
            model,
            args.vlayer,
            args.vbeta,
            args.vvar,
            ensemble=args.ensemble,
            progbar=False
        )

        img = np.array(image)
        vmap = cv2.resize(np.array(vmap), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"{args.output_path}/{image_id}", vmap * 255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BiomedCLIP Adapter Saliency Map Generator')
    parser.add_argument('--input-path', required=True, default="data/input_images", type=str, help='path to the images')
    parser.add_argument('--output-path', required=True, default="saliency_map_outputs", type=str, help='path to the output')
    parser.add_argument('--val-path', type=str, default="data/val_images", help='path to the validation set for hyperparameter optimization')
    parser.add_argument('--vbeta', type=float, default=0.1)
    parser.add_argument('--vvar', type=float, default=1.0)
    parser.add_argument('--vlayer', type=int, default=7)
    parser.add_argument('--tbeta', type=float, default=0.3)
    parser.add_argument('--tvar', type=float, default=1)
    parser.add_argument('--tlayer', type=int, default=9)
    parser.add_argument('--model-name', type=str, default="BiomedCLIP", help="Which CLIP model to use")
    parser.add_argument('--device', type=str, default="cuda", help="Device to run the model on")
    parser.add_argument('--ensemble', action='store_true', help="Whether to use text ensemble or not")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--json-path', type=str, default=None, help="Path to the JSON file containing the text prompts")
    parser.add_argument('--reproduce', action='store_true', help="Load text prompts from JSON file")
    parser.add_argument('--hyper-opt', action='store_true', help="Whether to optimize hyperparameters or not")
    parser.add_argument('--text-prompt', type=str, default=None, help="Text prompt to use for all images (overrides input/JSON)")

    # Adapter-specific options
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Base BiomedCLIP checkpoint (dir) or .pt file. If None, use HF model.')
    parser.add_argument('--adapter-checkpoint', type=str, default=None, help='Adapter training checkpoint (.pt) with model_state_dict')
    parser.add_argument('--adapter-layers', nargs='+', type=int, default=[9, 10, 11], help='Vision layers to inject adapters')
    parser.add_argument('--rank-k', type=int, default=64, help='SVD rank k for adapter')
    parser.add_argument('--gamma-init', type=float, default=0.1, help='Initial gamma for adapter fusion')
    parser.add_argument('--adapter-dropout', type=float, default=0.1, help='Adapter dropout')

    args = parser.parse_args()
    main(args)

    print("Saliency Map Generation Done!")
