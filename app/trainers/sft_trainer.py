"""
SFT (Supervised Fine-Tuning) trainer for RFT cold-start warmup.

Loads JSONL produced by generate_rft_data.py and fine-tunes the base model
for one epoch so that DAPO starts from a model that already knows the correct
output format and produces grounded reasoning traces.

Each JSONL record:
    {
        "image_b64":        str,   # base64-encoded JPEG of the chart image
        "question":         str,
        "completion":       str,   # the full model output to supervise on
        "label":            str,
        "consistency_score": float
    }
"""

import base64
import io
import json
import sys
from pathlib import Path
from typing import List, Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.prompts import SYSTEM_PROMPT
from data.preprocessing import process_image_for_model


def load_rft_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _decode_image(record: Dict):
    """Decode base64 image stored in JSONL record."""
    img_bytes = base64.b64decode(record["image_b64"])
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


class RFTSFTTrainer:
    """
    One-epoch SFT trainer that runs before DAPO.

    The training signal is the full (prompt + completion) sequence.  We do
    not mask the prompt — for a 1-epoch cold start this is fine and avoids
    the complexity of finding the exact split boundary in the tokenised input.
    """

    def __init__(
        self,
        model,
        processor,
        rft_path: str,
        config,
    ):
        self.model = model
        self.processor = processor
        self.records = load_rft_jsonl(rft_path)
        self.config = config
        self.lr = 2e-5
        self.image_min_pixels = getattr(config, "image_min_pixels", 4 * 28 * 28)
        self.image_max_pixels = getattr(config, "image_max_pixels", 320 * 28 * 28)
        self.image_resample = getattr(config, "image_resample", "lanczos")

    # ------------------------------------------------------------------
    def _build_inputs(self, record: Dict):
        """Build tokenised inputs for one SFT example."""
        try:
            image = _decode_image(record)
        except Exception:
            return None
        image = process_image_for_model(
            image,
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
            resample=self.image_resample,
        )
        if image is None:
            return None

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": record["question"]},
                ],
            },
            {"role": "assistant", "content": record["completion"]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        return inputs

    # ------------------------------------------------------------------
    def train(self):
        import random

        print(f"[SFT] Starting 1-epoch warmup on {len(self.records)} examples")

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.lr, weight_decay=0.01)
        scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=max(1, len(self.records) // 10),
        )

        self.model.train()
        total_loss = 0.0
        n_steps = 0
        skipped = 0

        shuffled = list(self.records)
        random.shuffle(shuffled)

        device = next(self.model.parameters()).device

        for step, record in enumerate(shuffled):
            try:
                inputs = self._build_inputs(record)
            except Exception as e:
                print(f"  [SFT WARN] step {step}: input build failed — {e}")
                skipped += 1
                continue

            if inputs is None:
                skipped += 1
                continue

            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = inputs["input_ids"].clone()
            # Ignore padding tokens in the loss (guard: pad_token_id can be None)
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100

            try:
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
            except Exception as e:
                print(f"  [SFT WARN] step {step}: forward failed — {e}")
                skipped += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_steps += 1

            if (step + 1) % 50 == 0:
                avg = total_loss / n_steps
                print(f"  [SFT] step {step + 1}/{len(shuffled)} | "
                      f"loss={avg:.4f} | skipped={skipped}")

        avg_loss = total_loss / max(n_steps, 1)
        print(f"[SFT] Done. avg_loss={avg_loss:.4f} | "
              f"trained={n_steps} | skipped={skipped}")

    # ------------------------------------------------------------------
    def save(self, output_dir: str):
        """Save LoRA adapter weights after SFT."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"[SFT] Saved to {output_dir}")
