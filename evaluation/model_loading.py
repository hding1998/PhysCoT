# -*- coding: utf-8 -*-
"""
model_loading.py — 统一的模型/分词器加载（评测、探针、DPO 采样共用）
====================================================================
解决的问题：train_sft.py 保存的是 PEFT adapter 目录（只有 adapter_config.json + adapter 权重），
`AutoModelForCausalLM.from_pretrained(adapter_dir)` 会直接失败。本模块自动识别：
  * HF 完整模型目录 / Hub 名称  → 直接加载；
  * PEFT adapter 目录           → 读 adapter_config.json 的 base_model_name_or_path，
                                  加载基座后挂载 adapter，并（默认）merge_and_unload 以加速推理。
另提供 merge_lora()：把 adapter 合并导出为完整模型目录（vLLM 推理 / 边缘部署需要）。
"""
import os
import json


def is_adapter_dir(path):
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))


def base_model_of(adapter_dir):
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as f:
        return json.load(f)["base_model_name_or_path"]


def load_tokenizer(path, padding_side="left"):
    from transformers import AutoTokenizer
    src = path
    if is_adapter_dir(path) and not os.path.exists(os.path.join(path, "tokenizer_config.json")):
        src = base_model_of(path)
    tok = AutoTokenizer.from_pretrained(src, padding_side=padding_side)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(path, dtype=None, device_map="auto", merge=True, trainable=False):
    """返回 (model, tokenizer)。adapter 目录自动挂载基座。"""
    import torch
    from transformers import AutoModelForCausalLM
    dtype = dtype or torch.bfloat16
    tok = load_tokenizer(path)
    if is_adapter_dir(path):
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            base_model_of(path), torch_dtype=dtype, device_map=device_map)
        model = PeftModel.from_pretrained(base, path, is_trainable=trainable)
        if merge and not trainable:
            model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map=device_map)
    return model, tok


def merge_lora(adapter_dir, out_dir, dtype=None):
    """adapter → 完整权重目录（供 vLLM / GGUF 导出 / 7B 对照复用）。"""
    import torch
    model, tok = load_causal_lm(adapter_dir, dtype=dtype or torch.bfloat16,
                                device_map="cpu", merge=True)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    return out_dir


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="合并 LoRA adapter 为完整模型目录")
    ap.add_argument("--adapter", required=True, help="如 runs/sft_v1/final")
    ap.add_argument("--out", required=True, help="如 runs/sft_v1/final-merged")
    a = ap.parse_args()
    print("merged ->", merge_lora(a.adapter, a.out))
