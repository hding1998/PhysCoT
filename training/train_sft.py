# -*- coding: utf-8 -*-
"""
train_sft.py — 阶段1：机理链 SFT（QLoRA / LoRA，2×3080-20GB）
================================================================
用法（实验机上，02_代码框架/ 目录下）：
  python training/train_sft.py --data datasets/sft_train.jsonl \
      --model Qwen/Qwen2.5-1.5B-Instruct --out runs/sft_v1 --epochs 3 --curriculum
  # bitsandbytes 不可用（如 Windows 原生装不上）时：加 --no_4bit（bf16 LoRA，1.5B 约 8GB 显存）

要点（v3，2026-09-21 实验机修订）：
  * 适配 TRL ≥0.17：使用 SFTConfig(max_length, completion_only_loss=True) + prompt/completion
    数据格式，loss 只算 completion（链是学习目标；旧版 DataCollatorForCompletionOnlyLM 已弃用）；
  * 默认 max_length=6144：SimBench-urban 样本 total max≈5019 token（monitor_cap=32 实测），4096 会截断 answer 阶段
    （20GB 卡上 1.5B QLoRA seq 4096 bs 2 约 10–12GB；先跑 05_环境/verify_env.py 的 [E] 长度审计）；
  * 课程模式 --curriculum：混入 build_curriculum.py 生成的单步样本（阶段掩码的数据侧实现）；
  * 训练结束保存 adapter 到 <out>/final，并（默认）合并导出 <out>/final-merged 供 vLLM 评测。
"""
import os
import json
import argparse

import torch
import torch.nn.functional as F


class ChunkedCESFTTrainer:
    """SFTTrainer 子类（工厂式混入）：分块交叉熵，解决全词表 logits 显存爆炸。

    背景：Qwen2.5 vocab=151936，seq≈5k 时 bf16 logits≈1.7GB、fp32 上转≈3.4GB、
    log_softmax 反向图再 ≈3.4GB，vocab 段合计 >10GB，20GB 卡训不动 1.5B。
    本实现绕过 lm_head 的全序列物化：只跑 decoder 拿 hidden，按 chunk 过 lm_head
    并对每个 chunk 立即反向（retain_graph 保 trunk 图），任一时刻只有一块
    chunk 的 logits 图存活（默认 chunk=1024 → fp32 峰值 ~0.6GB）。
    语义与 ForCausalLMLoss 一致：label 左移一位、-100 忽略、按 num_items_in_batch 归一。
    """

    CHUNK = 1024

    @staticmethod
    def _unwrap(model):
        """按类名逐层解包（不能用 hasattr 探测：peft 的 __getattr__ 转发 +
        PreTrainedModel.base_model 属性在末端返回自身，会死循环——本机实测卡死点）。"""
        m = model
        for _ in range(8):
            cn = type(m).__name__
            if cn in ("DataParallel", "DistributedDataParallel"):
                m = m.module
            elif cn.startswith("PeftModel"):
                m = m.base_model                     # PeftModelForCausalLM → LoraModel
            elif cn == "LoraModel":
                m = m.model                          # LoraModel → Qwen2ForCausalLM
            elif hasattr(m, "lm_head") and hasattr(m, "model"):
                return m.model, m.lm_head            # decoder, lm_head
            else:
                break
        raise RuntimeError(f"无法解包模型结构: {type(model).__name__} → {type(m).__name__}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
        from torch.utils.checkpoint import checkpoint

        decoder, lm_head = self._unwrap(model)
        hidden = decoder(input_ids=inputs["input_ids"],
                         attention_mask=inputs.get("attention_mask"),
                         use_cache=False).last_hidden_state
        shift_hidden = hidden[:, :-1]
        shift_labels = labels[:, 1:]
        if num_items_in_batch is None:
            n_tok = float(shift_labels.ne(-100).sum().item())
        else:
            n_tok = float(num_items_in_batch.item() if torch.is_tensor(num_items_in_batch)
                          else num_items_in_batch)
        n_tok = max(n_tok, 1.0)
        vocab = lm_head.weight.shape[0]

        def _chunk_ce(h, lb):
            lg = lm_head(h)                               # bf16 [B,c,V]
            return F.cross_entropy(lg.float().view(-1, vocab), lb.reshape(-1),
                                   ignore_index=-100, reduction="sum") / n_tok

        # 每块用 checkpoint 包装：前向不保留 chunk 的 fp32 logits 图，
        # 反向时逐块重算并立即释放 —— 任一时刻只有一块 (~1GB 级) 存活。
        loss = hidden.sum() * 0.0
        T = shift_hidden.size(1)
        for s in range(0, T, self.CHUNK):
            loss = loss + checkpoint(_chunk_ce,
                                     shift_hidden[:, s:s + self.CHUNK].contiguous(),
                                     shift_labels[:, s:s + self.CHUNK].contiguous(),
                                     use_reentrant=False)
        return (loss, None) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=2, help="per-device micro batch")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=6144)   # SimBench-urban total max≈5019（cap32实测），6144 留 22% 余量
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--curriculum", action="store_true", help="混合单步课程样本")
    ap.add_argument("--curriculum_ratio", type=float, default=0.3)
    ap.add_argument("--no_4bit", action="store_true", help="不用 bitsandbytes 4bit（bf16 LoRA）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_merge", action="store_true", help="训练后不导出合并权重")
    ap.add_argument("--max_samples", type=int, default=None, help="调试用：只取前 N 条")
    ap.add_argument("--no_chunk_ce", action="store_true",
                    help="关闭分块交叉熵（默认开启：vocab 151936 × seq 5k 的 logits 链 >10GB，20GB 卡必开）")
    ap.add_argument("--ce_chunk", type=int, default=1024, help="分块 CE 的序列块长")
    ap.add_argument("--cpu", action="store_true", help="CPU 冒烟（自动关闭 4bit/bf16）")
    ap.add_argument("--local_rank", type=int, default=-1)
    args = ap.parse_args()
    if args.cpu:
        args.no_4bit = True

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dev = {"": "cpu"} if args.cpu else {"": 0}
    load_kw = dict(torch_dtype=torch.float32 if args.cpu else torch.bfloat16, device_map=dev)
    if not args.no_4bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.max_samples:
        rows = rows[:args.max_samples]
    n_full = len(rows)
    if args.curriculum:
        cur_path = args.data.replace(".jsonl", "_curriculum.jsonl")
        if os.path.exists(cur_path):
            cur = [json.loads(l) for l in open(cur_path, encoding="utf-8")]
            n_cur = int(n_full * args.curriculum_ratio / (1 - args.curriculum_ratio))
            rows = rows + cur[:n_cur]
            print(f"课程混合: 全链 {n_full} + 单步 {min(len(cur), n_cur)}")
        else:
            print(f"[warn] 未找到课程样本 {cur_path}，仅用全链样本")
    # prompt/completion 格式 → TRL 自动只对 completion 计 loss（completion_only_loss=True）
    ds = Dataset.from_list([{"prompt": r["prompt"] + "\n", "completion": r["completion"]}
                            for r in rows]).shuffle(seed=args.seed)

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=not args.cpu,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        report_to="none",
        dataloader_num_workers=0 if args.cpu else 2,
        use_cpu=args.cpu,
        max_length=args.max_len,
        completion_only_loss=True,
        packing=False,
        seed=args.seed,
    )
    ChunkedCESFTTrainer.CHUNK = args.ce_chunk
    trainer_cls = SFTTrainer if args.no_chunk_ce else \
        type("ChunkedSFT", (ChunkedCESFTTrainer, SFTTrainer), {})
    trainer = trainer_cls(model=model, args=cfg, train_dataset=ds, processing_class=tok)

    # 冒烟核对：completion-only loss 是否生效（prompt 段 label 应为 -100）
    try:
        b = next(iter(trainer.get_train_dataloader()))
        lab = b["labels"][0]
        n_ign = int((lab == -100).sum()); n_tot = int(lab.numel())
        print(f"[check] 首个 batch labels: 忽略(prompt/pad) {n_ign}/{n_tot} token —— 应明显 >0")
    except Exception as e:                     # noqa: BLE001
        print("[check] 跳过 label 检查:", e)

    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    print("SFT done ->", final)

    if not args.no_merge:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation"))
        from model_loading import merge_lora
        merged = merge_lora(final, os.path.join(args.out, "final-merged"),
                            dtype=torch.float32 if args.cpu else torch.bfloat16)
        print("merged ->", merged)


if __name__ == "__main__":
    main()
