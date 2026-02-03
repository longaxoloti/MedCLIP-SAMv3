#!/usr/bin/env python3
"""
Quick Start Guide for Phase 2: Training & Evaluation

This script demonstrates how to run baseline evaluation and training scripts.
"""

import os
import subprocess
import sys
from pathlib import Path


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def run_command(cmd: str, description: str):
    """Run a command and print output."""
    print(f"\n→ {description}")
    print(f"  Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running command: {e}")
        return False


def main():
    """Main quickstart guide."""
    
    print_header("Phase 2: Training & Evaluation Quick Start")
    
    # 1. Setup paths
    print("\n1. Setting up paths...")
    
    dataset_root = os.path.expanduser("~/projects/MedCLIP-SAMv3/data")
    project_root = os.path.expanduser("~/projects/MedCLIP-SAMv3")
    results_dir = os.path.expanduser("~/projects/MedCLIP-SAMv3/phase2_results")
    
    print(f"   Dataset root: {dataset_root}")
    print(f"   Project root: {project_root}")
    print(f"   Results dir: {results_dir}")
    
    if not os.path.exists(dataset_root):
        print(f"\n✗ Dataset directory not found: {dataset_root}")
        print("  Please ensure the data directory exists with the 4 datasets:")
        print("  - brain_tumors/")
        print("  - breast_tumors/")
        print("  - lung_CT/")
        print("  - lung_Xray/")
        return False
    
    os.makedirs(results_dir, exist_ok=True)
    
    # 2. Baseline Evaluation
    print_header("STEP 1: Baseline Evaluation (No Adapters)")
    
    print("""
This step evaluates the pre-trained BiomedCLIP model WITHOUT any fine-tuning
to establish a baseline performance for comparison.

Output:
  - baseline_summary.csv: Summary metrics for all datasets
  - baseline_<dataset>_detailed.json: Detailed results per dataset
    """)
    
    baseline_cmd = f"""
cd {project_root} && python biomedclip_finetuning/evaluate_baseline.py \\
    --dataset-root {dataset_root} \\
    --output-dir {results_dir}/baseline_results \\
    --device cuda:0 \\
    --split test
    """
    
    print("Run baseline evaluation with:")
    print(f"  python biomedclip_finetuning/evaluate_baseline.py \\")
    print(f"      --dataset-root {dataset_root} \\")
    print(f"      --output-dir {results_dir}/baseline_results \\")
    print(f"      --device cuda:0 \\")
    print(f"      --split test")
    
    # 3. Training
    print_header("STEP 2: Train Frequency Adapters")
    
    print("""
This step trains the frequency adapters on each dataset using DHN-NCE loss.

Configuration:
  - Adapter layers: [9, 10, 11] (top 3 layers of ViT encoder)
  - Rank k: 64 (SVD rank)
  - Learning rate: 1e-4
  - Epochs: 50
  - Batch size: 32

Output:
  - checkpoint_best.pt: Best model checkpoint
  - checkpoint_epoch_*.pt: Epoch-wise checkpoints
  - training_history.json: Loss curves
  - training.log: Detailed training log
    """)
    
    datasets = ['brain_tumors', 'breast_tumors', 'lung_CT', 'lung_Xray']
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\nDataset {i}/{len(datasets)}: {dataset}")
        print(f"  python biomedclip_finetuning/train_frequency_adapter.py \\")
        print(f"      --dataset-root {dataset_root} \\")
        print(f"      --dataset {dataset} \\")
        print(f"      --output-dir {results_dir}/trained_adapters \\")
        print(f"      --epochs 50 \\")
        print(f"      --batch-size 32 \\")
        print(f"      --device cuda:0")
    
    # 4. Configuration Options
    print_header("CONFIGURATION OPTIONS")
    
    print("""
Baseline Evaluation:
  --dataset-root       : Root directory with datasets (required)
  --output-dir         : Output directory (default: ./baseline_results)
  --device             : Device to use (default: cuda:0)
  --split              : Data split [test|val|train] (default: test)

Training:
  --dataset-root       : Root directory with datasets (required)
  --dataset            : Dataset name (default: brain_tumors)
  --output-dir         : Output directory (default: ./checkpoints)
  --epochs             : Number of epochs (default: 50)
  --batch-size         : Batch size (default: 32)
  --lr                 : Learning rate (default: 1e-4)
  --weight-decay       : Weight decay (default: 1e-6)
  --device             : Device to use (default: cuda:0)
  --adapter-layers     : Layer indices for injection (default: 9 10 11)
  --rank-k             : SVD rank (default: 64)
  --gamma-init         : Initial gamma (default: 0.1)
  --num-workers        : Data loading workers (default: 4)
  --max-train-samples  : Max training samples (for debugging)
  --max-val-samples    : Max validation samples (for debugging)
    """)
    
    # 5. Recommended Training Schedules
    print_header("RECOMMENDED TRAINING STRATEGIES")
    
    print("""
FAST (Quick verification - 6-8 hours on A100):
  python biomedclip_finetuning/train_frequency_adapter.py \\
      --dataset-root ~/projects/MedCLIP-SAMv3/data \\
      --dataset brain_tumors \\
      --epochs 10 \\
      --batch-size 64 \\
      --device cuda:0

BALANCED (Production - 24-48 hours on A100):
  python biomedclip_finetuning/train_frequency_adapter.py \\
      --dataset-root ~/projects/MedCLIP-SAMv3/data \\
      --dataset brain_tumors \\
      --epochs 50 \\
      --batch-size 32 \\
      --device cuda:0

THOROUGH (Research - 48-72 hours on A100):
  python biomedclip_finetuning/train_frequency_adapter.py \\
      --dataset-root ~/projects/MedCLIP-SAMv3/data \\
      --dataset brain_tumors \\
      --epochs 100 \\
      --batch-size 16 \\
      --lr 5e-5 \\
      --device cuda:0
    """)
    
    # 6. Parallel Training on Multiple GPUs
    print_header("PARALLEL TRAINING (Multiple Datasets)")
    
    print("""
You can train on multiple datasets in parallel using different GPUs:

# GPU 0: Brain Tumors
python biomedclip_finetuning/train_frequency_adapter.py \\
    --dataset-root ~/projects/MedCLIP-SAMv3/data \\
    --dataset brain_tumors \\
    --output-dir ./checkpoints \\
    --epochs 50 \\
    --device cuda:0 &

# GPU 1: Breast Tumors
python biomedclip_finetuning/train_frequency_adapter.py \\
    --dataset-root ~/projects/MedCLIP-SAMv3/data \\
    --dataset breast_tumors \\
    --output-dir ./checkpoints \\
    --epochs 50 \\
    --device cuda:1 &

# GPU 2: Lung CT
python biomedclip_finetuning/train_frequency_adapter.py \\
    --dataset-root ~/projects/MedCLIP-SAMv3/data \\
    --dataset lung_CT \\
    --output-dir ./checkpoints \\
    --epochs 50 \\
    --device cuda:2 &

# GPU 3: Lung Xray
python biomedclip_finetuning/train_frequency_adapter.py \\
    --dataset-root ~/projects/MedCLIP-SAMv3/data \\
    --dataset lung_Xray \\
    --output-dir ./checkpoints \\
    --epochs 50 \\
    --device cuda:3 &

# Wait for all to complete
wait
    """)
    
    # 7. Expected Results
    print_header("EXPECTED RESULTS & OUTPUTS")
    
    print("""
After baseline evaluation, you should see:
  ✓ baseline_summary.csv
    - Dice coefficients for each dataset
    - IoU scores
    - Statistical summaries (mean ± std)
  
  ✓ baseline_<dataset>_detailed.json
    - Per-image metrics
    - Detailed statistics

After training, in each checkpoint directory:
  ✓ checkpoint_best.pt
    - Best model weights (lowest validation loss)
  
  ✓ checkpoint_epoch_*.pt
    - Weights at each epoch (for analysis)
  
  ✓ training_history.json
    - Epoch-wise train/val loss curves
    - Useful for visualization
  
  ✓ training.log
    - Detailed log of training process
    - Any warnings or errors
    """)
    
    # 8. Visualization
    print_header("VISUALIZING RESULTS")
    
    print("""
After training, you can plot loss curves:

import json
import matplotlib.pyplot as plt

# Load training history
with open('checkpoints/brain_tumors/2024*/training_history.json') as f:
    history = json.load(f)

plt.figure(figsize=(10, 6))
plt.plot(history['epoch'], history['train_loss'], label='Train')
plt.plot(history['epoch'], history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    """)
    
    # 9. Troubleshooting
    print_header("TROUBLESHOOTING")
    
    print("""
1. CUDA Out of Memory:
   - Reduce batch size: --batch-size 16
   - Reduce number of workers: --num-workers 2
   - Use CPU: --device cpu (slower)

2. HuggingFace Model Download Issues:
   - Check internet connection
   - Manually download: huggingface-cli download chuhac/BiomedCLIP-vit-bert-hf
   - Use --device cpu to download without GPU

3. Dataset Not Found:
   - Verify directory structure matches expected format
   - Check that data/brain_tumors, data/breast_tumors, etc. exist
   - Verify train_images, train_masks, val_images, val_masks subdirs exist

4. Low GPU Memory but have multiple GPUs:
   - Use --device cuda:1 to switch GPUs
   - Train datasets in parallel on different GPUs

5. Training is too slow:
   - Increase batch size (if memory allows)
   - Use --num-workers 8 (increase data loading workers)
   - Check GPU utilization: nvidia-smi
    """)
    
    # 10. Next Steps
    print_header("NEXT STEPS (Phase 3)")
    
    print("""
After Phase 2 training completes:

1. Evaluate adapted models on test sets
2. Compare with baseline results
3. Run ablation studies:
   - Different adapter layers
   - Different SVD ranks (k=32, 64, 128)
   - Different gamma initializations
   - Different training epochs
4. Verify no catastrophic forgetting on unseen diseases
5. Integration with M2IB + LG-SR + SAM pipeline
    """)
    
    print_header("READY TO START!")
    
    print(f"""
Your environment is ready. To begin:

1. Activate environment:
   source /home/long/venv/zeroshot/bin/activate

2. Run baseline evaluation:
   cd {project_root}
   python biomedclip_finetuning/evaluate_baseline.py \\
       --dataset-root {dataset_root} \\
       --output-dir {results_dir}/baseline_results \\
       --device cuda:0

3. Start training:
   python biomedclip_finetuning/train_frequency_adapter.py \\
       --dataset-root {dataset_root} \\
       --dataset brain_tumors \\
       --output-dir {results_dir}/trained_adapters \\
       --epochs 50 \\
       --batch-size 32 \\
       --device cuda:0

Good luck! 🚀
    """)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
