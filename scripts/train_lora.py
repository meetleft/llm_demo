"""QLoRA fine-tuning entry point for data/*.jsonl."""
from __future__ import annotations
import argparse, os
from settings import DEFAULT_MODEL, configure_huggingface

configure_huggingface()

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--train-file", default="data/train.jsonl")
    p.add_argument("--valid-file", default="data/valid.jsonl")
    p.add_argument("--output-dir", default="outputs/qwen25-1.5b-novel-lora")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--max-steps", type=int, default=-1, help="调试时可设为 1；默认按 epochs 训练")
    a = p.parse_args()
    if not os.path.exists(a.train_file):
        raise SystemExit(f"训练文件不存在：{a.train_file}；请先复制 data/train.example.jsonl")
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    files = {"train": a.train_file}
    if os.path.exists(a.valid_file): files["validation"] = a.valid_file
    ds = load_dataset("json", data_files=files)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model_kwargs = dict(quantization_config=quant, device_map="auto", trust_remote_code=True, dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.model, **model_kwargs)
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], task_type="CAUSAL_LM")
    train_args = SFTConfig(
        output_dir=a.output_dir, num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=16, gradient_checkpointing=True,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=10, save_steps=10, eval_steps=10,
        eval_strategy="steps" if "validation" in ds else "no",
        save_total_limit=2, fp16=True, bf16=False, report_to="none",
        max_length=a.max_seq_length, packing=False,
        dataset_num_proc=1, gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    kwargs = dict(model=model, args=train_args, train_dataset=ds["train"], peft_config=lora, processing_class=tok)
    if "validation" in ds: kwargs["eval_dataset"] = ds["validation"]
    trainer = SFTTrainer(**kwargs)
    trainer.train(); trainer.save_model(a.output_dir); tok.save_pretrained(a.output_dir)
    print(f"LoRA adapter 已保存到：{a.output_dir}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
