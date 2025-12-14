本项目基于论文代码思路实现基因型→表型预测，并在原有实现上**彻底修复并完善**以下关键点：

1. **严格消除数据泄露（No Leakage）**
   - phenotype 的 `StandardScaler` **在每个 fold 内仅使用训练集 fit**，再 transform val/test。
   - SNP 重要性筛选（MI/MIC）也**仅在训练集**上计算，然后应用到 val/test。

2. **Padding 安全的显式物理位置编码**
   - 位置向量 padding 部分不再用 0，而是用 **-1**（避免与真实最小位置冲突）。
   - 模型中对 padding 位置进行 mask：padding 位的 `pos_emb` 会被置零，从而不会污染输入。

3. **Masked Attention Pooling（解决 padding 污染注意力汇聚）**
   - `AttnPool1d` 支持 mask，padding 部分不会参与 softmax。
   - mask 会通过 CNN 的 3 次 MaxPool 近似下采样，使 pooling 时 mask 长度与 CNN 输出长度一致。

4. **显式区分 missing vs padding**
   - 计算 `missing_mask`（来自 1v 编码中 -1 的位置）。
   - 模型加入可学习的 `missing_emb`，仅在 missing 且非 padding 的位置叠加，使 missing 不再与 padding（全零）混淆。

5. **工程化改进**
   - EarlyStopping checkpoint 改为保存 dict（epoch/val_loss/state_dict），并支持按 fold 不同路径保存。
   - 加入随机种子与 cuDNN deterministic 设置，提升可复现性。

## 目录结构

```
base/
├── environment.yml
├── README.md
├── train.py
├── data_process/
│   ├── __init__.py
│   └── encode.py
└── model/
    ├── __init__.py
    ├── early_stop.py
    └── model_1v_depth.py
```

## 运行方法

1) 准备数据
- `data/dpc_data_100SW.csv`
- `data/dpc_data_100SW.map`

2) 训练

在 `base/` 目录下运行：

```bash
python train.py
```

你可以在 `train.py` 顶部的 `TrainConfig` 修改参数（k=1000、batch_size、epochs 等）。

## 关于 MIC

- 论文的 MIC 通常使用 MINE 算法（例如 `minepy`）。本项目默认使用 **MI（mutual_info_regression）** 作为稳定可用的替代。
- 若你安装了 `minepy`，可将 `TrainConfig.selector_method` 改为 `'mic'`，将自动启用 MIC。

