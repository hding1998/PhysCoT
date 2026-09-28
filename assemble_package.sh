#!/usr/bin/env bash
# assemble_package.sh — G6 打包组装（在 D: 盘构建完整交付包）
# 用法: bash assemble_package.sh
# 产出: /d/physcot_pack/PhysCoT/ + MANIFEST.md + 校验报告
set -u
ROOT="c:/Users/Administrator/Desktop/因果推理论文设计"
SRC="$ROOT/02_代码框架"
PACK="/d/physcot_pack/PhysCoT"
COLD="/d/physcot_cold/datasets"

echo "== 0. 清理旧包并建目录"
rm -rf "$PACK"
mkdir -p "$PACK"

echo "== 1. 代码框架（排除 __pycache__/.pytest_cache）"
mkdir -p "$PACK/02_代码框架"
for d in configs data_engine datasets evaluation training results runs; do
  cp -r "$SRC/$d" "$PACK/02_代码框架/$d"
done
cp "$SRC/build_datasets.py" "$PACK/02_代码框架/" 2>/dev/null
find "$PACK" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find "$PACK" -name "*.pyc" -delete 2>/dev/null

echo "== 2. 冷数据集回迁（D: 缓存 → 包内 datasets/）"
for f in "$COLD"/*.jsonl; do
  cp "$f" "$PACK/02_代码框架/datasets/"
done

echo "== 3. 源数据（01_数据）与文档（04_实验文档）"
cp -r "$ROOT/01_数据" "$PACK/01_数据"
cp -r "$ROOT/04_实验文档" "$PACK/04_实验文档"
cp "$ROOT/论文设计方案_PhysCoT配电网小模型因果推理后训练.md" "$PACK/" 2>/dev/null

echo "== 4. 环境定义"
mkdir -p "$PACK/05_环境"
cp "$ROOT/05_环境/requirements.txt" "$ROOT/05_环境/environment.yml" "$ROOT/05_环境/部署说明.md" "$PACK/05_环境/" 2>/dev/null

echo "== 5. 校验：adapter 基座指针抽查 3 个"
python - <<'EOF'
import json, os, random
PACK = "/d/physcot_pack/PhysCoT"
adapters = []
for root, _, files in os.walk(os.path.join(PACK, "02_代码框架/runs")):
    if "adapter_config.json" in files:
        adapters.append(os.path.join(root, "adapter_config.json"))
random.seed(0)
for p in random.sample(adapters, min(3, len(adapters))):
    base = json.load(open(p, encoding="utf-8"))["base_model_name_or_path"]
    # 基座存在性（包内 runs 或本机 D:/hf_cache）
    in_pack = os.path.exists(os.path.join(PACK, "02_代码框架", base.replace("/", os.sep)))
    on_disk = os.path.exists(base)
    print(f"  {os.path.relpath(p, PACK)} -> {base} | 包内={in_pack} 本机={on_disk}")
print(f"adapter 总数: {len(adapters)}")
EOF

echo "== 6. 校验：merged 权重缺失清单（可从 adapter 重建）"
python - <<'EOF'
import os
PACK = "/d/physcot_pack/PhysCoT"
runs = os.path.join(PACK, "02_代码框架/runs")
missing, present = [], []
for d in sorted(os.listdir(runs)):
    rd = os.path.join(runs, d)
    if not os.path.isdir(rd): continue
    has_adapter = os.path.exists(os.path.join(rd, "final/adapter_model.safetensors"))
    has_merged = os.path.exists(os.path.join(rd, "final-merged/model.safetensors"))
    if has_merged: present.append(d)
    elif has_adapter: missing.append(d)
print(f"  merged 在位: {len(present)} 个: {present}")
print(f"  仅 adapter（重建法: python evaluation/model_loading.py --adapter runs/<d>/final --out runs/<d>/final-merged）: {missing}")
EOF

echo "== 7. 大小统计"
du -sh "$PACK"/* 2>/dev/null
du -sh "$PACK"
echo "PACK_DONE -> $PACK"
