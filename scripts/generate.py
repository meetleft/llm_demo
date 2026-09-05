"""Generate Chinese novel text with the base model or a LoRA adapter."""
from __future__ import annotations
import argparse
import os
import sys
from settings import DEFAULT_ADAPTER, DEFAULT_MODEL, configure_huggingface

configure_huggingface()

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prompt", nargs="?", help="创作要求；不填则交互输入")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=DEFAULT_ADAPTER, help="LoRA adapter 目录；默认自动使用训练结果")
    p.add_argument("--base-only", action="store_true", help="只使用基础模型，不加载 LoRA")
    p.add_argument("--max-new-tokens", type=int, default=800)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-p", type=float, default=0.9)
    a = p.parse_args()
    prompt = a.prompt or input("创作要求：").strip()
    if not prompt: raise SystemExit("创作要求不能为空")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not os.path.isdir(a.model) and os.environ.get("HF_ENDPOINT"):
        print(f"使用 Hugging Face 镜像：{os.environ['HF_ENDPOINT']}", file=sys.stderr)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs = {"dtype": dtype, "device_map": "auto" if torch.cuda.is_available() else None}
    if torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, **kwargs)
    except Exception as exc:
        if "huggingface.co" in str(exc) or "SSL" in str(exc) or "MaxRetryError" in str(exc):
            raise SystemExit(
                "模型下载失败：当前网络无法稳定连接 Hugging Face。\n"
                "请设置 HF_ENDPOINT 镜像重试，或先下载到本地目录后用 --model 本地路径。\n"
                "示例：$env:HF_ENDPOINT='https://hf-mirror.com'"
            ) from exc
        raise
    adapter = None if a.base_only else a.adapter
    if adapter:
        from peft import PeftModel
        print(f"加载小说 LoRA：{adapter}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, adapter)
    messages = [{"role":"system","content":"你是中文长篇小说写作助手。遵守人物设定和世界观，推动剧情，避免复读。"},{"role":"user","content":prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=a.temperature, top_p=a.top_p, repetition_penalty=1.08, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
