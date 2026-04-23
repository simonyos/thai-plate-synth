"""Zero-shot VLM labeling of scraped real Thai plates.

Uses Qwen2.5-VL-3B-Instruct — small enough for 8GB VRAM (bf16, single image
at a time), strong on Asian-language OCR. The output is pseudo-ground-truth:
we trust the VLM enough to use its predictions as targets for downstream
student-model distillation / RL reward, but not enough to skip spot-check
verification on a random sample before scaling.

Output JSONL schema (one line per image):
    {
        "image": "<basename>",
        "registration": "<predicted plate registration string>",
        "province": "<predicted province name or empty>",
        "confidence": "<high|medium|low|unknown>",
        "raw_output": "<full VLM response>",
        "model": "Qwen/Qwen2.5-VL-3B-Instruct"
    }

Resumable: an existing output file is read, and already-labeled images are
skipped. This lets a long run be interrupted and continued without re-work.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROMPT = """This image shows a Thai vehicle license plate. Read only the text printed on the plate itself.

Respond with EXACTLY these three lines and nothing else:
REGISTRATION: <the top line — Thai consonants and Arabic digits only, no spaces>
PROVINCE: <the bottom line — Thai province name, or empty if not visible>
CONFIDENCE: <high | medium | low>

Example:
REGISTRATION: 6กพ7414
PROVINCE: กรุงเทพมหานคร
CONFIDENCE: high

If no plate is visible or the image is not a Thai plate, respond:
REGISTRATION:
PROVINCE:
CONFIDENCE: low
"""

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def _parse(raw: str) -> dict:
    def _grab(key: str) -> str:
        m = re.search(rf"^{key}:\s*(.*?)\s*$", raw, flags=re.MULTILINE)
        return (m.group(1).strip() if m else "")

    # VLM occasionally puts CONFIDENCE on the same line as PROVINCE or
    # REGISTRATION — split-and-strip once we have the value.
    def _strip_spillover(v: str) -> str:
        for key in ("CONFIDENCE:", "PROVINCE:", "REGISTRATION:"):
            if key in v:
                v = v.split(key, 1)[0].strip()
        return v

    return {
        "registration": _strip_spillover(_grab("REGISTRATION")),
        "province": _strip_spillover(_grab("PROVINCE")),
        "confidence": _strip_spillover(_grab("CONFIDENCE")).lower() or "unknown",
    }


def _load_model():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"loading {MODEL_ID} …", flush=True)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # Hard cap on image tokens: default allows up to ~2100 vision tokens per
    # image which is too much for 8GB VRAM on scene-level inputs (these caused
    # the OOM cascade in the first pass). 768²/28² ≈ 750 tokens comfortably
    # fits alongside model weights + KV cache for any resolution.
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=768 * 28 * 28,
    )
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    return model, processor


def _predict(model, processor, img_path: Path) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    gen = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0]


def label(
    images_dir: Path,
    out_path: Path,
    limit: int | None = None,
    resume: bool = True,
) -> None:
    files = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if limit:
        files = files[:limit]

    done: set[str] = set()
    if resume and out_path.is_file():
        # Keep only successful records; prior error records are retried.
        kept: list[str] = []
        n_retry = 0
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("confidence") == "error":
                n_retry += 1
                continue
            done.add(rec["image"])
            kept.append(line)
        out_path.write_text("\n".join(kept) + ("\n" if kept else ""))
        print(f"resume: {len(done)} already labeled, {n_retry} prior errors will be retried", flush=True)

    remaining = [p for p in files if p.name not in done]
    print(f"labeling {len(remaining)} images ({len(files) - len(remaining)} already done)", flush=True)

    if not remaining:
        return

    model, processor = _load_model()

    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        t0 = time.time()
        for i, p in enumerate(remaining):
            # Proactive cache clear every 50 images prevents the fragmentation-
            # induced OOM cascade seen in the first full run at image ~1559.
            if i > 0 and i % 50 == 0:
                torch.cuda.empty_cache()
            try:
                raw = _predict(model, processor, p)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                try:
                    raw = _predict(model, processor, p)
                except Exception as e2:
                    err = f"OOM-retry-failed: {str(e2)[:90]}"
                    record = {
                        "image": p.name, "registration": "", "province": "",
                        "confidence": "error", "raw_output": err, "model": MODEL_ID,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"[{i+1}/{len(remaining)}] ERR {p.name}: {err}", flush=True)
                    continue
            except Exception as e:
                raw = ""
                err = str(e)[:120]
                record = {
                    "image": p.name,
                    "registration": "",
                    "province": "",
                    "confidence": "error",
                    "raw_output": err,
                    "model": MODEL_ID,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{i+1}/{len(remaining)}] ERR {p.name}: {err}", flush=True)
                continue

            parsed = _parse(raw)
            record = {
                "image": p.name,
                **parsed,
                "raw_output": raw.strip(),
                "model": MODEL_ID,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            if (i + 1) % 10 == 0 or i < 5:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(remaining) - (i + 1)) / max(rate, 1e-6)
                reg = parsed["registration"] or "(empty)"
                conf = parsed["confidence"]
                print(
                    f"[{i+1}/{len(remaining)}] {p.name[:40]:<40} → {reg[:20]:<20} conf={conf} "
                    f"(rate={rate:.2f}/s, eta={eta/60:.1f}m)",
                    flush=True,
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=Path("data/real_scrape/roboflow/images"))
    ap.add_argument("--out", type=Path, default=Path("data/real_scrape/roboflow/vlm_labels.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="stop after N images (pilot runs)")
    ap.add_argument("--no-resume", action="store_true", help="ignore existing output file")
    args = ap.parse_args()
    label(args.images, args.out, args.limit, resume=not args.no_resume)


if __name__ == "__main__":
    main()
