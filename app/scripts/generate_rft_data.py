#!/usr/bin/env python3
"""
Generate RFT (Reject-Sampling Fine-Tuning) data from a trained checkpoint.

Runs the checkpoint on a sample of training examples and keeps only the
completions that are both:
  (a) answer-correct  — normalize_answer(pred) == normalize_answer(gt)
  (b) table-consistent — at least `--min-consistency` fraction of the values
                          the model put in <table> also appear in its reasoning

Saves one JSONL record per kept completion:
    {
        "dataset_name": str,
        "row_idx":      int,   <- index in the shuffled sample, used for
                                  image re-loading during SFT
        "question":     str,
        "completion":   str,
        "label":        str,
        "consistency_score": float
    }

Usage:
    python scripts/generate_rft_data.py \\
        --checkpoint outputs/dapo_hcpc_v2/run_xxx/checkpoints/step_300 \\
        --output rft_data.jsonl \\
        --n-samples 3000 \\
        --n-rollouts 8 \\
        --temperature 1.3
"""

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import load_dataset
from PIL import Image

from data.preprocessing import process_image_for_model
from data.prompts import SYSTEM_PROMPT, format_conversation
from models.loader import load_model_with_checkpoint
from utils.parsing import normalize_answer
from rewards.consistency_reward import compute_consistency_reward


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Path to trained LoRA checkpoint directory")
    p.add_argument("--output", default="rft_data.jsonl",
                   help="Output JSONL file path")
    p.add_argument("--dataset", default="sanchit97/chart-rvr-grpo-train",
                   help="HF training dataset to sample from")
    p.add_argument("--n-samples", type=int, default=3000,
                   help="Number of training examples to run rollouts on")
    p.add_argument("--n-rollouts", type=int, default=8,
                   help="Generation rollouts per example")
    p.add_argument("--temperature", type=float, default=1.3,
                   help="Sampling temperature (higher → more diverse rollouts)")
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--min-consistency", type=float, default=0.5,
                   help="Min fraction of table values that must appear in reasoning")
    p.add_argument("--cache-dir", default="./cache")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    return p.parse_args()


def _get_first(example, keys):
    for k in keys:
        if k in example and example[k] is not None:
            return example[k]
    return None


def _image_to_b64(pil_image: Image.Image) -> str:
    """Encode a PIL image to base64 JPEG string for JSONL storage."""
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _parse_answer(completion: str) -> str:
    m = _ANSWER_RE.search(completion)
    return m.group(1).strip() if m else ""


def main():
    args = parse_args()

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model, processor = load_model_with_checkpoint(
        checkpoint_path=args.checkpoint,
        base_model_name=args.base_model,
        device_map="auto",
        cache_dir=args.cache_dir,
    )
    model.eval()
    device = next(model.parameters()).device

    # ── Load dataset ────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train", cache_dir=args.cache_dir)
    dataset = dataset.shuffle(seed=args.seed)
    n = min(args.n_samples, len(dataset))
    dataset = dataset.select(range(n))
    print(f"Sampled {n} examples")

    # ── Generate ─────────────────────────────────────────────────────────────
    kept = []
    total_rollouts = 0
    total_correct = 0
    total_consistent = 0

    for idx in range(len(dataset)):
        example = dataset[idx]

        question = str(
            _get_first(example, ["query", "question", "prompt", "input"]) or ""
        )
        gt_label = str(
            _get_first(example, ["label", "answer", "answers", "output"]) or ""
        )
        image = example.get("image")

        if not question or not gt_label or image is None:
            continue
        if not isinstance(image, Image.Image):
            continue

        try:
            image_orig = image  # keep original PIL before any processing
            image = process_image_for_model(image)
            messages = format_conversation(question)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            ).to(device)
        except Exception as e:
            print(f"  [WARN] idx={idx}: input error — {e}")
            continue

        prompt_len = inputs["input_ids"].shape[1]

        for _ in range(args.n_rollouts):
            total_rollouts += 1
            try:
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        do_sample=True,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )
                completion = processor.tokenizer.decode(
                    out_ids[0][prompt_len:], skip_special_tokens=True
                )
            except Exception as e:
                print(f"  [WARN] idx={idx}: generation error — {e}")
                continue

            # (a) answer correctness
            pred = normalize_answer(_parse_answer(completion))
            gt = normalize_answer(gt_label)
            if not pred or pred != gt:
                continue
            total_correct += 1

            # (b) table–reasoning consistency
            score = compute_consistency_reward(completion)
            if score < args.min_consistency:
                continue
            total_consistent += 1

            kept.append({
                "image_b64": _image_to_b64(image_orig),
                "question": question,
                "completion": completion,
                "label": gt_label,
                "consistency_score": round(score, 4),
            })

        if (idx + 1) % 100 == 0:
            pct_kept = 100 * len(kept) / max(total_rollouts, 1)
            print(
                f"  [{idx + 1}/{len(dataset)}] rollouts={total_rollouts} "
                f"correct={total_correct} consistent={total_consistent} "
                f"kept={len(kept)} ({pct_kept:.1f}%)"
            )

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\nDone. Kept {len(kept)} / {total_rollouts} rollouts "
          f"({100 * len(kept) / max(total_rollouts, 1):.1f}%)")
    print(f"  Answer-correct  : {total_correct} "
          f"({100 * total_correct / max(total_rollouts, 1):.1f}%)")
    print(f"  Also consistent : {total_consistent} "
          f"({100 * total_consistent / max(total_correct, 1):.1f}% of correct)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for record in kept:
            f.write(json.dumps(record) + "\n")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
