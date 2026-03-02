# -*- coding: utf-8 -*-
"""
HCPC-RLVR Kaggle Training Notebook
====================================
Kaggle-specific version of colab_file.py.
Tested on: Kaggle + H100 (80 GB), 9-hour session limit.
 
Experiments available:
  dapo_hcpc_v2      ← flagship (DAPO + HCPC-v2)
  dapo_baseline     ← DAPO without HCPC
  dapo_hcpc         ← DAPO + original HCPC (ablation)
  grpo_baseline / grpo_hcpc / nsr_baseline / nsr_hcpc / ...
 
Cell 1 — Clone repo
"""
 
import os
 
REPO_URL = "https://github.com/potate4/chartrl.git"
BRANCH   = "fix/kaggle-hcpc"
WORK_DIR = "/kaggle/working/chartrl"
 
if os.path.exists(WORK_DIR):
    print("Repo exists — pulling latest…")
    os.chdir(WORK_DIR)
    os.system(f"git pull origin {BRANCH}")
else:
    print("Cloning…")
    os.system(f"git clone -b {BRANCH} {REPO_URL} {WORK_DIR}")
    os.chdir(WORK_DIR)
 
os.chdir(f"{WORK_DIR}/app")
print("Working directory:", os.getcwd())
 
 
# ── Cell 2 — Install dependencies ─────────────────────────────────────────
"""
Run this cell once per session.
"""
 
os.system("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
os.system("pip install -q 'transformers>=4.49.0' 'accelerate>=0.34.0' 'peft>=0.13.0'")
os.system("pip install -q datasets pillow tqdm wandb")
os.system("pip install -q sentence-transformers bitsandbytes")
os.system("pip install -q qwen-vl-utils")
os.system("pip install -q 'trl>=0.12.0'")
os.system("pip install -U trl")
 
import torch
print(f"GPU available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name      : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM          : {vram:.1f} GB")
 
 
# ── Cell 3 — Restore checkpoint from Kaggle Dataset (optional) ────────────
"""
If you have a previous run saved as a Kaggle Dataset or Output, restore it:
    DATASET_INPUT = "/kaggle/input/chartrl-checkpoints"
    restore_from_kaggle(DATASET_INPUT)
"""
 
import shutil
from pathlib import Path
 
 
def restore_from_kaggle(
    source_dir: str,
    local_dir: str = "/kaggle/working/chartrl/app/outputs",
):
    """Copy checkpoint directories from a Kaggle input dataset into outputs/."""
    src = Path(source_dir)
    dst = Path(local_dir)
    if not src.exists():
        print(f"No checkpoint dataset found at {source_dir}")
        return
 
    for exp_dir in src.iterdir():
        if not exp_dir.is_dir():
            continue
        for run_dir in exp_dir.glob("run_*"):
            dest_run = dst / exp_dir.name / run_dir.name
            if dest_run.exists():
                print(f"  Skipping {run_dir.name} (already exists)")
                continue
            print(f"  Restoring {exp_dir.name}/{run_dir.name} …")
            shutil.copytree(run_dir, dest_run)
    print("Restore complete.")
 
 
# Uncomment to restore:
# restore_from_kaggle("/kaggle/input/your-checkpoint-dataset")
 
 
# ── Cell 4 — Verify imports ───────────────────────────────────────────────
 
import sys
sys.path.insert(0, ".")
 
from configs.experiment import EXPERIMENTS, list_experiments
from data import load_training_dataset
 
print("Imports OK.\n")
list_experiments()
 
 
# ── Cell 5 — Quick smoke test (10 samples, 1 epoch) ───────────────────────
"""
Verify the full pipeline works before committing to a full run.
"""
 
os.system(
    "python scripts/train.py "
    "--experiment dapo_hcpc_v2 "
    "--subset-size 10 "
    "--num-epochs 1 "
    "--no-wandb"
)
 
 
# ── Cell 6 — Full training: DAPO + HCPC-v2 (flagship) ────────────────────
"""
Recommended for a single Kaggle 9-hour H100 session.
  subset_size=2000, 1 epoch, batch=2, 4 rollouts ≈ 6–8 hours on H100.
"""
 
os.system(
    "python scripts/train.py "
    "--experiment dapo_hcpc_v2 "
    "--subset-size 2000 "
    "--num-epochs 1 "
    "--no-wandb "
    "--batch-size 2 "
    "--num-generations 4"
)
 
 
# ── Cell 7 — Resume from checkpoint ──────────────────────────────────────
"""
To resume from a specific TRL checkpoint:
"""
 
RESUME_CKPT = "/kaggle/working/chartrl/app/outputs/dapo_hcpc_v2/run_YYYYMMDD_HHMMSS/trl_output/checkpoint-NNN"
 
os.system(
    f"python scripts/train.py "
    f"--resume-from {RESUME_CKPT} "
    f"--experiment dapo_hcpc_v2 "
    f"--subset-size 2000 "
    f"--num-epochs 1 "
    f"--no-wandb "
    f"--batch-size 2 "
    f"--num-generations 4"
)
 
 
# ── Cell 8 — List runs ────────────────────────────────────────────────────
 
os.system("python scripts/train.py --experiment dapo_hcpc_v2 --list-runs")
 
 
# ── Cell 9 — GPU monitor ──────────────────────────────────────────────────
 
os.system("nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv")