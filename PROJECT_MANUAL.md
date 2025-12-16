# 深度学习项目手册（工程版）

> 项目名称：`dpcformer`（见 `environment.yml`）  
> 项目类型：基因型（SNP）→ 表型（phenotype）预测（回归）  
> 代码入口：`train.py`（当前仓库仅提供训练与评估流程，未提供独立推理脚本）

---

# 1. 项目概述

## 1.1 项目简介

### 1.1.1 项目类型

- **基因型 → 表型预测（Genotype-to-Phenotype, G2P）回归任务**：输入为个体 SNP 基因型序列（按染色体组织），输出为连续数值表型（`phenotype`）。
- 领域上属于 **生物信息/基因组选择（Genomic Prediction）** 场景的 **监督学习回归**。

### 1.1.2 解决的核心问题

在给定：
- **CSV**：每个样本的 `genotype`（字符串形式的 SNP token 序列）与 `phenotype`（连续表型）
- **MAP**：SNP 的染色体归属与物理位置（`position`）

的情况下，构建模型预测表型，并在工程上重点解决：

1) **严格避免数据泄露（No Leakage）**
- 训练脚本对每个 fold：`StandardScaler` 仅在训练集 `fit`，再对 val/test `transform`（`train.py`）。
- SNP 重要性筛选（MI/MIC）也只在训练集上计算，再应用到 val/test（`train.py`）。

2) **不同染色体 SNP 数不足 K 时的 padding 污染**
- 每条染色体都选 Top-K SNP；若实际 SNP 数不足 K，则对尾部 padding。
- 模型使用 `pad_mask` 对 padding 位进行显式 masking，防止位置 embedding、注意力汇聚被污染（`model/model_1v_depth.py`）。

3) **显式区分 missing vs padding**
- 缺失基因型在 8v 编码中可能表现为全 0（与 padding 的全 0 容易混淆）。
- 训练管线额外构造 `missing_mask`（由 1v 编码中 `-1` 生成），模型引入可学习的 `missing_emb` 用于缺失位点建模（`train.py` + `model/model_1v_depth.py`）。

### 1.1.3 算法/模型类型

#### 特征选择（Feature Selection）
- **互信息（MI）**：使用 `sklearn.feature_selection.mutual_info_regression`，并将 SNP 离散码视为离散特征（`data_process/encode.py`）。
- **MIC（可选）**：当 `TrainConfig.selector_method='mic'` 时尝试使用 `minepy` 计算 MIC；若 `minepy` 不可用则回退 MI（`data_process/encode.py`）。

#### 深度模型（Deep Model）
训练脚本实际使用的模型为：
- `PhenotypePredictor_HCA_Pos`（`model/model_1v_depth.py`）

其核心结构：
- **每条染色体独立 CNN 编码器**（1D 卷积残差块 + MaxPool 下采样）
- **Masked Attention Pooling**：对 CNN 输出序列做带 mask 的注意力汇聚成单个染色体 token
- **染色体级 Transformer Encoder**：建模染色体之间的交互
- **MLP 回归头**：输出单值表型预测
- **显式物理位置 embedding（MLP）**：由 MAP 的 `position` 归一化得到，并严格 mask 掉 padding 位
- **missing embedding**：仅对 missing 且非 padding 的位置叠加

> 备注：文件中保留了 `PhenotypePredictor_1v` 作为旧基线兼容实现，但训练脚本未使用它。

---

## 1.2 项目整体架构

### 1.2.1 Pipeline 总览（训练/评估）

```mermaid
flowchart TD
  A[输入数据<br/>CSV: genotype+phenotype<br/>MAP: chromosome+position] --> B[load_map<br/>解析染色体切片与物理位置]
  A --> C[read_and_encode_all<br/>分块读取 CSV]
  C --> D1[8v 编码<br/>每 SNP 8 维 one-hot]
  C --> D2[1v 编码<br/>每 SNP 离散码]
  D1 --> E[划分数据：test + train/val]
  D2 --> E
  E --> F[每 fold: KFold 划分 train/val]
  F --> G1[StandardScaler 仅 train fit<br/>再 transform val/test]
  F --> G2[每染色体 MI/MIC 仅 train 计算<br/>选择 Top-K 并按位置排序]
  G2 --> H[构造 phys_pos/pad_mask<br/>pos padding=-1]
  D1 --> I[build_split_tensors<br/>按 selected indices 取子集并 padding 到 K]
  D2 --> I
  I --> J[构造 DataLoader<br/>TensorDataset(X, missing_mask, y)]
  H --> K[构造模型 PhenotypePredictor_HCA_Pos<br/>传入 phys_pos + pad_mask]
  K --> L[训练循环<br/>Adam + MSE + ReduceLROnPlateau]
  L --> M[验证评估<br/>MSE + Pearson + Spearman]
  M --> N[EarlyStopping 保存 checkpoint]
  N --> O[加载最佳 checkpoint]
  O --> P[测试集评估与保存 state_dict]
  P --> Q[打印 CV 汇总与均值相关性]
```

### 1.2.2 模块依赖关系

- `train.py`
  - 依赖 `data_process/encode.py`：编码与特征选择
  - 依赖 `model/model_1v_depth.py`：模型定义
  - 依赖 `model/early_stop.py`：早停与 checkpoint 存取
  - 依赖第三方：PyTorch、NumPy、Pandas、SciPy、Scikit-learn

- `data_process/encode.py`
  - 依赖第三方：NumPy、Pandas、Scikit-learn
  - 可选依赖：`minepy`（用于 MIC）

- `model/model_1v_depth.py`
  - 依赖第三方：PyTorch

- `model/early_stop.py`
  - 依赖第三方：PyTorch、标准库

### 1.2.3 核心目录结构（代码视角）

```text
real/
├── environment.yml
├── README.md
├── train.py
├── data/
│   └── (放置数据：dpc_data_100SW.csv / dpc_data_100SW.map)
├── data_process/
│   ├── __init__.py
│   └── encode.py
├── model/
│   ├── __init__.py
│   ├── early_stop.py
│   └── model_1v_depth.py
└── output/
    ├── loss_png/         (预留目录，当前代码未生成图)
    └── saved_models/     (checkpoint/state_dict 输出目录)
```

---

# 2. 环境与运行指南

## 2.1 环境依赖

### 2.1.1 Python 版本
- `Python 3.9`（`environment.yml`）

### 2.1.2 依赖库（以 `environment.yml` 为准）

Conda（主要科学计算与数据处理）：
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `tqdm`

Pip（深度学习）：
- `torch==2.3.1`
- `torchvision==0.18.1`
- `torchaudio==2.3.1`

可选依赖：
- `minepy`：当 `TrainConfig.selector_method='mic'` 时启用 MIC；未安装会自动回退 MI。

### 2.1.3 GPU/CPU 要求
- 代码支持 CPU 运行；若 `torch.cuda.is_available()` 为 True，则使用 GPU `cuda:0`。
- 内存注意：项目会把全量编码后的数据合并到内存（`one_hot_8v`、`geno_1v`），数据较大时 RAM 压力显著。

### 2.1.4 CUDA/cuDNN 版本
- 项目未显式固定 CUDA/cuDNN 版本；以你安装的 PyTorch 构建版本为准。
- 脚本设置了：
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  用于增强可复现性，但可能降低性能。

---

## 2.2 项目配置

项目没有独立的配置文件（如 YAML）；全部配置集中在 `train.py` 的 `TrainConfig` 数据类中。

### 2.2.1 配置入口
- `TrainConfig`（`train.py`）

### 2.2.2 关键配置项说明（逐项）

| 配置项 | 类型 | 默认值 | 作用 |
|---|---:|---:|---|
| `csv_path` | `str` | `data/dpc_data_100SW.csv` | 输入 CSV 路径（必须包含 `genotype` 与 `phenotype` 列） |
| `map_path` | `str` | `data/dpc_data_100SW.map` | 输入 MAP 路径（定义 SNP 顺序/染色体边界/物理位置） |
| `chunksize` | `int` | `100` | CSV 分块读取大小（行数），控制读入过程峰值内存 |
| `selected_snp_count` | `int` | `1000` | 每条染色体选择的 Top-K SNP 数（K） |
| `test_ratio` | `float` | `0.1` | 全数据中划分独立 test 集比例 |
| `n_splits` | `int` | `10` | train/val 上的 KFold 折数 |
| `batch_size` | `int` | `64` | 训练 batch size |
| `epochs` | `int` | `100` | 每 fold 最大训练 epoch |
| `lr` | `float` | `1e-3` | Adam 学习率 |
| `selector_method` | `str` | `mi` | 特征选择方法：`mi` 或 `mic`（可选依赖 `minepy`） |
| `seed` | `int` | `42` | 随机种子（shuffle、KFold 等） |
| `out_dir` | `str` | `output` | 输出目录（保存模型/预留图像） |

---

## 2.3 如何运行训练

### 2.3.1 准备环境

使用 conda 创建环境（推荐）：

```bash
conda env create -f environment.yml
conda activate dpcformer
```

### 2.3.2 准备数据集

按 `README.md` 约定放置到：
- `data/dpc_data_100SW.csv`
- `data/dpc_data_100SW.map`

#### CSV 格式要求（由代码推断）
- 分隔符：`\t`（脚本使用 `sep='\t'`）
- 必须包含列：
  - `genotype`：字符串，每行一个样本；内部以空格分隔 SNP token
  - `phenotype`：数值型表型
- 每行 genotype token 个数必须与 MAP 行数一致，否则报错（脚本会校验）。

#### MAP 格式要求（由代码推断）
- 分隔符：`\t`
- 无 header
- 列依次为：`chromosome, snp_id, map, position`
- `chromosome` 用于划分染色体边界；`position` 用于物理位置编码。

### 2.3.3 启动训练

```bash
python train.py
```

### 2.3.4 常见训练参数含义
- `selected_snp_count`：每条染色体选多少 SNP（K），影响输入长度与显存/速度。
- `n_splits`：KFold 次数；越大训练越慢但估计更稳定。
- `test_ratio`：独立测试集比例；越大测试更稳定但训练数据更少。
- `selector_method`：MI（默认）或 MIC（需要 minepy）。

### 2.3.5 GPU / AMP / DDP
- GPU：自动启用（如果可用）。
- AMP：当前未实现。
- DDP：当前未实现。

---

## 2.4 如何运行推理（inference）

### 2.4.1 当前仓库现状（重要）
- 项目当前 **没有独立推理脚本**（如 `infer.py`）。
- 推理过程只在训练脚本中以“评估”的形式出现（`evaluate()` 在 val/test 上做 forward）。

### 2.4.2 为什么独立推理不完整（必须理解的工程约束）
要对“新样本”做推理，除了模型权重外，还必须具有与训练一致的 **元数据**：
1) 每条染色体的 `selected_rel_indices`（Top-K SNP 的相对索引列表）
2) 与之匹配的 `phys_pos_tensor`（归一化位置，padding=-1）
3) 与之匹配的 `pad_mask_tensor`
4) 表型 `StandardScaler` 的参数（因为训练标签被标准化；推理输出需要反标准化到原量纲）

但当前代码只保存了：
- fold checkpoint（包含 `model_state_dict`）
- fold 的 state_dict（`.pth`）

**未保存**上述 (1)(2)(3)(4) 元数据，因此“跨进程/跨机器”的独立推理无法严格复现训练语义。

### 2.4.3 当前代码可执行的“推理”
- 在 `train.py` 内部：
  - `EarlyStopping.load_best(model)` 加载最佳权重
  - `evaluate(model, test_loader, device)` 在 test 上前向推理并计算指标

---

# 3. 项目目录结构说明

## 3.1 目录树

```text
real/
├── data/                      数据目录（需手动放置 CSV/MAP）
├── data_process/              数据解析/编码/特征选择
│   ├── __init__.py
│   └── encode.py
├── model/                     模型结构与训练辅助
│   ├── __init__.py
│   ├── early_stop.py
│   └── model_1v_depth.py
├── output/                    训练输出（自动生成）
│   ├── loss_png/              预留目录
│   └── saved_models/          checkpoint + state_dict
├── environment.yml            conda 环境定义
├── README.md                  项目说明
└── train.py                   训练入口脚本
```

## 3.2 各目录职责

- `data/`
  - 存放原始数据文件（CSV、MAP）。
  - 路径由 `TrainConfig.csv_path/map_path` 控制。

- `data_process/`
  - 负责将 CSV 中的 genotype 字符串解析为矩阵，并提供：
    - 8v 编码（模型输入特征）
    - 1v 编码（特征选择与 missing mask）
    - MI/MIC 选择器

- `model/`
  - 模型文件与训练辅助逻辑：
    - `model_1v_depth.py`：网络结构（CNN/AttentionPool/Transformer/MLP/pos/missing）
    - `early_stop.py`：早停与 checkpoint 管理

- `output/`
  - 训练产物目录：
    - `saved_models/`：checkpoint 与 state_dict
    - `loss_png/`：预留（可扩展为画 loss 曲线）

---

# 4. 模型架构与核心算法说明

## 4.1 模型文件解析（`model/model_1v_depth.py`）

### 4.1.1 模型总体输入输出

训练脚本构造的模型输入来自 `build_split_tensors()`：
- `x`: `(B, num_chromosomes, K, 8)` float32
- `missing_mask`: `(B, num_chromosomes, K)` bool

模型输出：
- `out`: `(B, 1)` float32（标准化表型空间）

### 4.1.2 `PhenotypePredictor_HCA_Pos`（主模型）

**设计目的**
- 同时建模：
  - SNP 局部相互作用（CNN）
  - 染色体级全局交互（Transformer）
  - SNP 物理位置的显式信息（position embedding）
  - missing 与 padding 的语义差异（missing embedding + mask）

**结构概述（模块级）**
1) 对每条染色体：
   - 位置 embedding（MLP）叠加到 SNP 特征（仅有效位）
   - missing embedding 叠加（仅 missing 且非 padding）
   - CNN 下采样与局部特征提取
   - Masked Attention Pooling 得到“染色体 token”
2) 将所有染色体 token 组成序列，输入 Transformer 编码
3) flatten 后通过 MLP 回归头输出表型

**前向传播逻辑（自然语言描述）**
- 遍历每条染色体 i：
  1) 从输入 `x` 取出该染色体的 `(B,K,8)` 特征；
  2) 根据 `pad_mask[i]` 得到该染色体有效位（所有样本共享）；
  3) 取 `phys_pos[i]` 的归一化位置，并对 padding 位用 0 替代以保持数值稳定；
  4) 位置经过 MLP 得到 `(B,K,8)` 的位置 embedding，并用 `pad_mask` 将 padding 位 embedding 置 0；
  5) 将位置 embedding 加到 SNP 特征上；
  6) 若提供 `missing_mask`，则对 missing 且非 padding 的位置叠加可学习 `missing_emb`；
  7) 转为 CNN 需要的 `(B,8,K)`，通过 3 层残差卷积块 + MaxPool 得到 `(B,Cc,Lc)`；
  8) 将 `pad_mask` 通过与 CNN 下采样对应的 maxpool 近似下采样为 `(B,Lc)`，作为 attention pooling 的 mask；
  9) 对 `(B,Lc,Cc)` 做带 mask 的注意力池化，得到染色体 token `(B,Cc)`；
- 所有染色体 token 堆叠为 `(B,num_chr,Cc)`，经过 Transformer 编码后 flatten，输入 MLP 输出 `(B,1)`。

**与其他模块的耦合关系**
- 与 `train.py` 强耦合：
  - 必须提供与选出的 Top-K SNP 完全一致的 `phys_pos_tensor/pad_mask_tensor`
  - `missing_mask` 必须由相同 indices 从 `geno_1v` 中生成
- 与 loss/optimizer 解耦：
  - 模型仅输出预测值；loss/optimizer 在 `train.py` 中定义

---

## 4.2 损失函数（loss）说明

训练与评估均使用 **均方误差（MSE）**：
- 训练：`torch.nn.MSELoss()`（默认 reduction=mean）
- 评估：`torch.nn.MSELoss(reduction='sum')` 再除以样本数得到 MSE

**输入/输出格式**
- `pred`: `(B,1)` float32
- `target`: `(B,1)` float32
- `loss`: 标量

**为什么选择该 loss（推断）**
- 表型为连续值 → 回归任务 → MSE 是标准目标函数。

---

## 4.3 优化器与训练策略

### 4.3.1 optimizer
- Adam：`torch.optim.Adam(model.parameters(), lr=cfg.lr)`

### 4.3.2 scheduler
- `ReduceLROnPlateau`：
  - 监控 `val_mse`
  - `factor=0.1, patience=10, min_lr=1e-6`

### 4.3.3 EarlyStopping
- `EarlyStopping(patience=20)`：监控 `val_mse`
- 保存 fold-specific checkpoint dict：`epoch/val_loss/model_state_dict`

### 4.3.4 AMP / 梯度累积 / DDP
当前代码未实现：
- AMP：无 `autocast/GradScaler`
- 梯度累积：无累积逻辑
- DDP：无 `DistributedDataParallel/DistributedSampler`

---

# 5. 数据处理（data pipeline）说明

## 5.1 数据集类 Dataset

项目未自定义 Dataset；使用 `torch.utils.data.TensorDataset`。

单个样本的训练输入由 `build_split_tensors()` 组装，`TensorDataset` 每次返回：
- `x`: `(num_chr,K,8)` float32
- `missing_mask`: `(num_chr,K)` bool
- `y`: `(1,)` float32（标准化后的表型）

## 5.2 DataLoader 说明

- `train_loader`：`shuffle=True`

- `val_loader/test_loader`：`shuffle=False`

- 默认 `num_workers=0`（单进程）

- 未启用 `pin_memory/persistent_workers`

  ***注：数据在训练前已全部转为 Tensor 并驻留内存，DataLoader 主要负责 batch 切分与 shuffle。***

## 5.3 数据格式说明

### 5.3.1 输入数据（CSV）
- `genotype`：每行一个样本，字符串 token 序列，token 用空格分隔
- `phenotype`：连续数值标签

### 5.3.2 SNP token 与缺失定义
- token 常见：`AA, AT, AC, ...`
- 缺失/不完整常见：`00, A0, 0A, ...`

### 5.3.3 编码定义

**1v 编码（离散码，用于特征选择与 missing）**
- 每 SNP → int16
- 缺失为 `-1`

**8v 编码（模型输入特征）**
- 每 SNP → 8 维：allele1(4维 one-hot) + allele2(4维 one-hot)
- 缺失 allele 映射为全 0（因此缺失 SNP 的 8v 也可能是全 0）
- 通过 `missing_mask` 区分缺失与 padding

---

# 6. 训练流程详解

训练入口为 `train.py` 的 `main()`。

## 6.1 数据读取与全量编码

1) 解析 MAP：
   - 得到 `chrom_ranges`：每条染色体在全 SNP 序列中的切片范围
   - 得到 `chrom_positions`：每条染色体每个 SNP 的物理位置数组
2) 分块读取 CSV：
   - 对每个 chunk：
     - `genotype_to_dataframe` 将 genotype 字符串拆成 token 矩阵
     - 8v 编码：`base_one_hot_encoding_8v_dif`（扁平）→ reshape 成 `(N,n_snps,8)`
     - 1v 编码：`base_one_hot_encoding_1v` 生成 `(N,n_snps)`
3) 合并所有 chunk 得到全量：
   - `one_hot_8v`、`geno_1v`、`phenotype_raw`

## 6.2 数据划分策略

1) 先划分独立测试集（test）：
   - 用随机种子打乱样本索引
   - 取前 `test_ratio` 作为 `test_ids`
   - 剩余作为 `train_val_ids`
2) 在 `train_val_ids` 上做 `KFold(n_splits, shuffle=True)`：
   - 每 fold 得到 `train_ids/val_ids`

## 6.3 每 fold 的训练关键点

### 6.3.1 表型标准化（StandardScaler）
- `scaler.fit_transform` 仅在 `train_ids` 的表型上执行
- `val/test` 仅用 `transform`

### 6.3.2 特征选择（每染色体 Top-K）
- 在每条染色体切片上：
  - 取 `geno_1v[train_ids, start:end]`（训练集）
  - 用 `MISelector(method=selector_method)` 计算每 SNP 与 `y_train` 的关联得分
  - 选择 Top-K（若不足 K 则实际小于 K）
  - 根据物理位置重新排序（保持由近到远的顺序）

### 6.3.3 构造位置张量与 padding mask
- `pad_mask`：长度 K，真实 SNP 为 True，padding 为 False
- `phys_pos`：将选中 SNP 的 position 归一化到 [0,1]；padding 位置填 -1

### 6.3.4 组装模型输入张量
对 train/val/test 分别执行：
- 从 `one_hot_8v` 中按染色体切片取出，再按 `selected_rel_indices` 选列，padding 到 `(N_split,K,8)`
- 从 `geno_1v` 中取出相同 loci，`== -1` 得到 missing，padding 到 `(N_split,K)`
- 堆叠所有染色体：
  - `X`: `(N_split,num_chr,K,8)`
  - `missing_mask`: `(N_split,num_chr,K)`

## 6.4 训练循环（逐步解析）

对每 epoch：
1) `model.train()`
2) 遍历 batch：
   - `out = model(x, missing_mask=miss)`
   - `loss = MSE(out, y)`
   - `loss.backward()`
   - `optimizer.step()`
3) 验证：
   - `evaluate`：计算 `val_mse/pearson/spearman`
4) scheduler：
   - `ReduceLROnPlateau.step(val_mse)`
5) early stopping：
   - `EarlyStopping(val_mse, model, epoch)` 保存更优模型
   - 若 patience 超限则停止

fold 结束：
- 加载最佳 checkpoint → test 集评估
- 保存 `model.state_dict()` 到 `output/saved_models/model_fold_{fold}.pth`

---

# 7. 推理流程说明

## 7.1 推理入口函数
- 当前推理入口等价于 `evaluate()`：
  - `out = model(x, missing_mask=miss)`

## 7.2 模型加载方法

两类输出文件：
1) `checkpoint_fold_{fold}.pt`：dict checkpoint（epoch/val_loss/model_state_dict）
2) `model_fold_{fold}.pth`：纯 `state_dict`

建议优先使用 checkpoint（更适合“加载最佳模型”语义）。

## 7.3 输入预处理（必须与训练一致）

要做独立推理，必须“复现训练时生成输入张量的全过程”，并且必须拿到训练期的元数据：
- `selected_rel_indices`（每染色体的 Top-K SNP 相对索引）
- `phys_pos_tensor/pad_mask_tensor`
- `StandardScaler` 参数（用于反标准化）

当前仓库未保存上述元数据；若要补齐推理能力，建议在训练时把它们序列化保存（例如 `output/saved_models/fold_{k}_meta.json` + `npz`）。

## 7.4 输出后处理
- 模型输出位于“标准化表型空间”；需要 `scaler.inverse_transform` 才能还原原始表型量纲。

---

# 8. 类与函数的详细文档（逐文件）

> 说明：本项目 Python 源码文件共 4 个（不含空 `__init__.py`）。本节按文件逐一列出：类/函数职责、参数与返回值、调用关系、形状约定。

## 8.1 文件：`train.py`

### Class: `TrainConfig`（数据类）

- 所在文件：`train.py`
- 作用：集中管理训练超参数与路径配置。
- 关键字段：见第 2.2.2 表格。
- 调用关系：`main()` 内实例化并使用。

### Function: `set_seed(seed: int) -> None`

- 所在文件：`train.py`
- 功能描述：设置 Python/NumPy/PyTorch 随机种子，并配置 cuDNN 确定性行为。
- 参数：
  - `seed: int`：随机种子
- 返回值：`None`
- 使用场景：训练启动时保证可复现性。
- 被哪些文件调用：仅 `train.py` 内部 `main()` 调用。

### Function: `load_map(map_path: str) -> Tuple[pd.DataFrame, List[Tuple[int,int]], List[np.ndarray]]`

- 所在文件：`train.py`
- 功能描述：读取 MAP 文件，按 chromosome 分组，生成：
  - 染色体在全 SNP 序列中的切片范围
  - 每条染色体的 position 数组
- 参数：
  - `map_path: str`：MAP 文件路径
- 返回值：
  - `map_df: pd.DataFrame`：包含 `chromosome/snp_id/map/position`
  - `ranges: List[Tuple[int,int]]`：每条染色体的 `(start,end)` 半开区间
  - `positions: List[np.ndarray]`：每条染色体 position 序列（float32）
- 使用场景：为染色体切片、位置编码提供基础元数据。
- 被哪些文件调用：仅 `train.py` 内部 `main()` 调用。

### Function: `read_and_encode_all(cfg: TrainConfig, n_snps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`

- 所在文件：`train.py`
- 功能描述：分块读取 CSV，编码 genotype 得到：
  - 8v one-hot（模型输入特征）
  - 1v 离散码（特征选择与 missing）
  - phenotype 原始值
- 参数：
  - `cfg: TrainConfig`：包含 `csv_path/chunksize`
  - `n_snps: int`：SNP 总数（MAP 行数）
- 返回值：
  - `one_hot_8v: np.ndarray`：shape `(N,n_snps,8)`，float32
  - `geno_1v: np.ndarray`：shape `(N,n_snps)`，int16
  - `phenotype_raw: np.ndarray`：shape `(N,)`，float32
- 使用场景：训练前全量读入与编码。
- 被哪些文件调用：仅 `train.py` 内部 `main()` 调用。
- 关键约束：
  - CSV 每行 genotype token 数必须等于 `n_snps`，否则抛 `ValueError`。

### Function: `select_snps_per_chromosome(...) -> Tuple[List[np.ndarray], torch.Tensor, torch.Tensor]`

- 所在文件：`train.py`
- 功能描述：
  - 在**训练集**上对每条染色体计算 MI/MIC 得分并选择 Top-K SNP
  - 同时构造模型必需的：
    - `phys_pos_tensor`: `(num_chr,K)` float32（padding=-1）
    - `pad_mask_tensor`: `(num_chr,K)` bool
- 参数（重点）：
  - `geno_1v: np.ndarray`：`(N,n_snps)`，离散码
  - `y_train: np.ndarray`：`(N_train,)`，标准化后的训练表型
  - `train_ids: np.ndarray`：训练样本索引
  - `chrom_ranges: List[Tuple[int,int]]`：染色体切片
  - `chrom_positions: List[np.ndarray]`：每染色体 position 数组
  - `k: int`：Top-K
  - `selector_method: str`：`mi` 或 `mic`
- 返回值：
  - `selected_rel_indices: List[np.ndarray]`：每条染色体的相对列索引（相对于该染色体切片）
  - `phys_pos_tensor: torch.Tensor`：`(num_chr,K)` float32
  - `pad_mask_tensor: torch.Tensor`：`(num_chr,K)` bool
- 使用场景：每个 fold 构造模型结构所依赖的 indices/位置/mask。
- 被哪些文件调用：仅 `train.py` 内部 `main()` 调用。
- 无泄露说明：
  - `X_chr` 使用 `geno_1v[train_ids, start:end]`，不包含 val/test。

### Function: `build_split_tensors(...) -> Tuple[torch.Tensor, torch.Tensor]`

- 所在文件：`train.py`
- 功能描述：对某个 split（train/val/test）按预选 indices 组装输入张量并生成 missing mask。
- 参数：
  - `one_hot_8v: np.ndarray`：`(N,n_snps,8)`
  - `geno_1v: np.ndarray`：`(N,n_snps)`
  - `sample_ids: np.ndarray`：该 split 的样本索引
  - `chrom_ranges: List[Tuple[int,int]]`
  - `selected_rel_indices: List[np.ndarray]`
  - `k: int`：padding 后长度
- 返回值：
  - `X: torch.Tensor`：`(N_split,num_chr,K,8)` float32
  - `missing_mask: torch.Tensor`：`(N_split,num_chr,K)` bool
- 使用场景：
  - 与 `PhenotypePredictor_HCA_Pos` 的输入强耦合（形状与语义必须一致）。
- 被哪些文件调用：
  - `train.py` 的 `main()`（对 train/val/test 各调用一次）。

### Function: `evaluate(model, loader, device) -> Tuple[float, float, float]`

- 所在文件：`train.py`
- 功能描述：对一个 DataLoader 做推理评估，输出：
  - MSE（按样本平均）
  - Pearson 相关
  - Spearman 相关
- 参数：
  - `model: torch.nn.Module`：需支持 `model(x, missing_mask=...)`
  - `loader: DataLoader`：batch 输出 `(x, miss, y)`
  - `device: torch.device`
- 返回值：
  - `mse: float`
  - `pearson: float`
  - `spearman: float`
- 使用场景：
  - 每 epoch 的验证评估
  - 每 fold 的测试评估
- 被哪些文件调用：仅 `train.py` 内部 `main()` 调用。

### Function: `main() -> None`

- 所在文件：`train.py`
- 功能描述：训练入口，完成数据读取、编码、划分、每 fold 训练/评估/保存、CV 汇总。
- 参数：无
- 返回值：`None`
- 被哪些文件调用：脚本入口 `if __name__ == '__main__': main()`

### Function: `compute_loss()`（现状说明）

- 本项目现状：**没有单独定义** `compute_loss()`；loss 逻辑直接写在训练循环中：
  - 训练：`criterion = torch.nn.MSELoss()`，`loss = criterion(out, y)`
  - 评估：`criterion = torch.nn.MSELoss(reduction='sum')`，`mse = loss_sum / N`
- 若你希望引入 `compute_loss(pred, target, reduction=...)`，建议将训练与评估的 loss 计算统一封装到 `train.py`，并由 `main()` 与 `evaluate()` 调用。

---

## 8.2 文件：`data_process/encode.py`

### Function: `genotype_to_dataframe(genotype_series: pd.Series) -> pd.DataFrame`

- 所在文件：`data_process/encode.py`
- 功能描述：将每行 genotype 字符串按空格拆分为 token 序列，得到二维表（样本 × SNP）。
- 参数：
  - `genotype_series: pd.Series`：每个元素为一个样本的 genotype 字符串
- 返回值：
  - `pd.DataFrame`：shape `(N,n_snps)`，每个单元格为一个 token（如 `AA`、`AT`、`00`）
- 使用场景：CSV 读取后，将字符串形式的 genotype 转为矩阵。
- 被哪些文件调用：`train.py` 的 `read_and_encode_all()`。

### Function: `_normalize_token_array(seq_all_df: pd.DataFrame) -> np.ndarray`

- 所在文件：`data_process/encode.py`
- 功能描述：统一 token 格式：
  - NaN → `00`
  - 转大写
  - 补齐为长度 2 的字符串（右侧补 `0`）
- 参数：
  - `seq_all_df: pd.DataFrame`
- 返回值：
  - `np.ndarray`：dtype 为 2 字符字符串（`<U2`）
- 使用场景：为 1v/8v 编码做一致预处理。
- 被哪些文件调用：
  - `base_one_hot_encoding_1v`
  - `base_one_hot_encoding_8v_dif`

### Function: `base_one_hot_encoding_1v(seq_all_df: pd.DataFrame) -> pd.DataFrame`

- 所在文件：`data_process/encode.py`
- 功能描述：将 SNP token 编码为离散整数码：
  - 常见二等位组合映射到 0..9
  - 缺失/不完整映射到 -1
- 参数：
  - `seq_all_df: pd.DataFrame`
- 返回值：
  - `pd.DataFrame`：shape `(N,n_snps)`，dtype int16
- 使用场景：
  - MI/MIC 特征选择输入
  - missing mask 构造（`== -1`）
- 被哪些文件调用：`train.py` 的 `read_and_encode_all()`。

### Function: `base_one_hot_encoding_8v_dif(seq_all_df: pd.DataFrame) -> pd.DataFrame`

- 所在文件：`data_process/encode.py`
- 功能描述：8v 编码（保留 allele 顺序）：
  - allele1: 4 维 one-hot（A/C/G/T）
  - allele2: 4 维 one-hot（A/C/G/T）
  - 拼接为 8 维
  - 缺失/未知 allele → 全 0（因此需要 `missing_mask` 区分）
- 参数：
  - `seq_all_df: pd.DataFrame`
- 返回值：
  - `pd.DataFrame`：shape `(N, n_snps*8)`，每 SNP 8 维被扁平拼接
- 使用场景：模型输入特征来源；在 `train.py` 中会 reshape 成 `(N,n_snps,8)`。
- 被哪些文件调用：`train.py` 的 `read_and_encode_all()`。

### Function: `sanitize_discrete_codes(X, missing_code: int = -1) -> np.ndarray`

- 所在文件：`data_process/encode.py`
- 功能描述：将缺失码 `-1` 映射到新的非负类别（max_code+1），使 `mutual_info_regression(discrete_features=True)` 更稳定。
- 参数：
  - `X: Union[np.ndarray, pd.DataFrame]`：2D 离散码矩阵
  - `missing_code: int`：默认 -1
- 返回值：
  - `np.ndarray`：缺失被替换后的矩阵
- 使用场景：MI 计算前的数据清洗。
- 被哪些文件调用：`MISelector.fit()`。

### Class: `MISelector`（sklearn 风格选择器）

- 所在文件：`data_process/encode.py`
- 作用：对 SNP 列打分并选择 Top-K。
- 主要方法：
  - `fit(X, y)`：
    - 若 `method='mic'` 且 `minepy` 可用则计算 MIC；否则计算 MI
    - 保存 `scores_` 与 `top_k_indices_`
  - `transform(X)`：返回 `X[:, top_k_indices_]`
- 参数与含义：
  - `k: int`：选多少列
  - `method: str`：`mi` 或 `mic`
  - `n_neighbors: int`：MI 估计器参数
  - `random_state`
  - `missing_code: int`：默认 -1
- 返回值：
  - `fit`：返回 `self`
  - `transform`：返回 `np.ndarray` 子矩阵
- 使用场景：在 `train.py` 的 `select_snps_per_chromosome()` 中，对每条染色体训练集片段进行特征选择。
- 被哪些文件调用：`train.py`。

### Class: `MICSelector`（兼容旧名）

- 所在文件：`data_process/encode.py`
- 作用：兼容旧类名；内部直接继承 `MISelector` 并发出弃用警告。

---

## 8.3 文件：`model/early_stop.py`

### Class: `EarlyStopCheckpoint`

- 所在文件：`model/early_stop.py`
- 作用：承载“最佳 checkpoint”的信息（val_loss/epoch/state_dict），供 `load_best` 返回。

### Class: `EarlyStopping`

- 所在文件：`model/early_stop.py`
- 作用：早停控制器；当验证损失在 `patience` 轮内未提升则停止训练，并在提升时保存 checkpoint。
- 关键参数：
  - `patience: int`：容忍轮数
  - `delta: float`：提升阈值
  - `path: str`：保存路径
  - `verbose: bool`：是否打印日志
- 关键方法：
  - `__call__(val_loss, model, epoch)`：更新早停状态，必要时保存 checkpoint
  - `_save_checkpoint(val_loss, model, epoch)`：保存 dict checkpoint
  - `load_best(model, map_location=None) -> EarlyStopCheckpoint`：加载保存的最佳权重到模型
- 使用场景：`train.py` 每个 fold 的训练循环中用于保存最优模型与控制提前停止。
- 被哪些文件调用：`train.py`。

---

## 8.4 文件：`model/model_1v_depth.py`

### Class: `PositionalEncoding`

- 所在文件：`model/model_1v_depth.py`
- 作用：Transformer 的 sin/cos 位置编码（序列下标意义，不是 MAP 物理位置）。
- forward 输入输出：
  - 输入 `x`: `(seq_len, batch, d_model)`
  - 输出：同 shape，`x + pe`
- 调用关系：被 `TransformerEncoderModel` 调用。

### Class: `TransformerEncoderModel`

- 所在文件：`model/model_1v_depth.py`
- 作用：封装 `nn.TransformerEncoder`，输入 `(B,S,C)` 输出同形状。
- forward 输入输出：
  - 输入 `x`: `(B,S,C)`
  - 可选 `src_key_padding_mask`: `(B,S)`（项目中未显式使用）
  - 输出：`(B,S,C)`
- 调用关系：
  - `PhenotypePredictor_1v` 使用它对 SNP 序列特征建模
  - `PhenotypePredictor_HCA_Pos` 使用它对染色体 token 序列建模

### Class: `PhysicalPositionEmbedding`

- 所在文件：`model/model_1v_depth.py`
- 作用：将归一化物理位置 `(B,L)` 映射到 embedding `(B,L,input_dim)`。
- forward 输入输出：
  - 输入 `pos`: `(B,L)` float
  - 输出 `emb`: `(B,L,input_dim)` float
- 调用关系：被 `PhenotypePredictor_HCA_Pos.forward` 调用，并在外部对 padding 位做 mask（置 0）。

### Class: `AttnPool1d`

- 所在文件：`model/model_1v_depth.py`
- 作用：序列注意力池化，支持 mask，输出 pooled token 与注意力权重。
- forward 输入输出：
  - 输入 `x`: `(B,L,C)`
  - 可选 `mask`: `(B,L)` bool（True=有效）
  - 输出：
    - `pooled`: `(B,C)`
    - `weights`: `(B,L)`
- 调用关系：被 `PhenotypePredictor_HCA_Pos.forward` 用于对每条染色体 CNN 输出池化。

### Class: `ResConvBlockLayer`

- 所在文件：`model/model_1v_depth.py`
- 作用：残差 1D 卷积块 + MaxPool 下采样 + Dropout。
- forward 输入输出（概念）：
  - 输入 `(B,C_in,L)` → 输出 `(B,C_out,L')`（由 maxpool 决定下采样）
- 调用关系：被 `ChromosomeCNN` 组合使用。

### Class: `ChromosomeCNN`

- 所在文件：`model/model_1v_depth.py`
- 作用：对单条染色体序列做 3 层残差卷积块下采样，提取局部特征。
- forward 输入输出（概念）：
  - 输入 `(B,input_dim,L)` → 输出 `(B,16,Lc)`
- 调用关系：
  - 被 `PhenotypePredictor_1v`、`PhenotypePredictor_HCA_Pos` 使用。

### Class: `PhenotypePredictor_1v`（基线/兼容）

- 所在文件：`model/model_1v_depth.py`
- 作用：旧基线模型（未在训练脚本中使用）。
- forward 输入输出（概念）：
  - 输入 `x`: `(B,num_chr,L,input_dim)`
  - 输出 `out`: `(B,1)`
- 调用关系：当前 `train.py` 未实例化该类。

### Class: `PhenotypePredictor_HCA_Pos`（主模型）

- 所在文件：`model/model_1v_depth.py`
- 作用：主训练模型，支持物理位置 embedding、masked attention pooling、missing embedding。
- forward 输入输出说明：
  - 输入 `x`: `(B,num_chr,L,input_dim)`（本项目 L=K，input_dim=8）
  - 输入 `missing_mask`（可选）: `(B,num_chr,L)` bool
  - 输出：
    - `out`: `(B,1)` float
    - 若 `return_attn=True`：返回 `(out, pool_attns)`
- 调用了哪些子模块：
  - `ChromosomeCNN`（每染色体）
  - `PhysicalPositionEmbedding`
  - `AttnPool1d`
  - `TransformerEncoderModel`
  - `MLP head`
- 在 `train.py` 中被如何使用：
  - 每 fold 构造一次，传入 `phys_pos_tensor/pad_mask_tensor`
  - 训练/评估调用：`model(x, missing_mask=miss)`

### Function: `print_output_shape(module, input, output)`（调试）

- 所在文件：`model/model_1v_depth.py`
- 功能描述：用于 forward hook 调试输出张量形状。
- 被哪些文件调用：仅该文件 `__main__` 调试段可能使用。

### Function: `register_hooks(model)`（调试）

- 所在文件：`model/model_1v_depth.py`
- 功能描述：遍历模型子模块注册 forward hook。
- 返回值：hook 列表（便于后续移除）。
- 被哪些文件调用：仅该文件 `__main__` 调试段可能使用。

---

## 8.5 文件：`data_process/__init__.py`、`model/__init__.py`

- 所在文件：`data_process/__init__.py`、`model/__init__.py`
- 作用：当前为空文件，仅用于声明包结构。
- 无类/函数定义。

---

# 9. 可视化与日志（如适用）

## 9.1 是否使用 TensorBoard / WandB / MLflow
- 当前代码未集成 TensorBoard/WandB/MLflow。

## 9.2 当前日志内容
训练过程中通过 `print()` 输出：
- `TrainMSE`
- `ValMSE`
- `ValPearson`
- `ValSpearman`
- `LR`
以及 fold/test 的汇总指标。

## 9.3 可视化脚本说明
- `output/loss_png/` 目录会被创建，但当前代码未生成任何图像文件。
- 若要补齐，可在训练循环中记录 `train_loss/val_mse` 并用 `matplotlib` 绘制保存。

---

# 10. 测试（如适用）

## 10.1 当前测试现状
- 当前仓库未提供单元测试/集成测试框架与用例。

## 10.2 建议的最小测试集（工程建议）
建议新增测试覆盖：
1) `data_process/encode.py`
   - token 规范化（大小写、NaN、长度）
   - 1v 编码缺失为 -1 的稳定性
   - 8v 编码 shape 与缺失全 0 行为
2) `train.py` 的 `build_split_tensors`
   - 形状 `(N_split,num_chr,K,8)` 与 missing mask 的对齐
3) `model/model_1v_depth.py`
   - `PhenotypePredictor_HCA_Pos` forward 输出 shape、无 NaN

---

# 11. 常见问题（FAQ）与错误排查

## 11.1 数据集加载报错怎么办

### 现象：`Genotype token count mismatch ...`
- 原因：CSV 每行 `genotype` token 数与 MAP 行数不一致。
- 排查：
  - CSV 是否使用 `\t` 分隔（代码按 `sep='\t'` 读取）
  - `genotype` 列是否存在、是否含有异常空格/缺失 token
  - MAP 是否与 CSV 对应同一套 SNP 顺序与数量

### 现象：`KeyError: 'genotype'` / `KeyError: 'phenotype'`
- 原因：CSV 列名与代码期望不一致。
- 解决：修改 CSV 列名或修改 `train.py` 读取列名。

## 11.2 GPU 内存不够怎么办

### 现象：`CUDA out of memory`
常见触发点：
- batch 太大
- K 太大（每条染色体输入长度）
- 模型中 CNN/Transformer 计算与中间激活占用大

解决建议（优先级从高到低）：
1) 降低 `TrainConfig.batch_size`
2) 降低 `TrainConfig.selected_snp_count`（K）
3) 改为 CPU 或更大显存 GPU
4) 工程改造：引入 AMP、减少中间缓存、改流式数据管线

## 11.3 checkpoint 不匹配怎么办

### 现象：`load_state_dict` size mismatch
- 原因：模型结构依赖 `num_chromosomes/K/卷积输出长度` 等；推理/恢复时的这些参数与训练时不一致会导致维度不匹配。
- 解决：
  - 使用与训练时完全一致的 `K` 与染色体数
  - 使用对应 fold 的 `phys_pos/pad_mask`（本项目每 fold 不同）
  - 优先用 `checkpoint_fold_{fold}.pt` 加载 `model_state_dict`

## 11.4 相关性指标异常（NaN）怎么排查
- 若预测或标签几乎为常数，`pearsonr/spearmanr` 可能返回 NaN。
- 排查：
  - 查看 val/test 的预测分布是否塌缩
  - 检查标准化流程是否正常（训练集样本数过少时也可能不稳定）

## 11.5 有哪些日志可以帮助排查
- epoch 级：`TrainMSE/ValMSE/LR`
- 指标级：`ValPearson/ValSpearman`
- fold 级：`TestMSE/Pearson/Spearman` 与各 fold 的方差

---

# 12. 未来扩展建议

## 12.1 模块解耦建议
- 将 `train.py` 拆分为：
  - `data_module.py`：数据解析/编码/split/dataloader
  - `trainer.py`：训练循环/评估/保存
  - `configs/`：支持 YAML/argparse，便于复现实验与开源使用

## 12.2 推理闭环建议（强烈建议）
为实现可用的独立推理，建议在训练时保存元数据：
- `selected_rel_indices`
- `phys_pos_tensor/pad_mask_tensor`
- `StandardScaler` 参数（mean/scale）
并提供 `infer.py`：
- 输入：新样本 CSV 或单条 genotype 字符串 + MAP
- 输出：原量纲表型预测 + 可选注意力权重

## 12.3 性能优化建议
- 数据侧：
  - 避免全量 `np.concatenate`，改为流式 Dataset 或 memory-map
  - `DataLoader(num_workers>0, pin_memory=True)`
- 训练侧：
  - AMP（`autocast` + `GradScaler`）
  - 梯度裁剪、weight decay
- 特征选择侧：
  - MI/MIC 并行化或缓存结果

## 12.4 更科学的训练策略（建议）
- 引入更系统的实验记录（seed、fold、K、selector_method、模型超参）
- 评估指标若更关注相关性，可考虑以 `ValPearson` 作为早停/调度器监控指标（需要你明确业务目标）

## 12.5 更易开源的工程化建议
- 提供 `requirements.txt`（或同时支持 conda/pip）
- 增加 CLI：`python train.py --csv ... --map ... --k ... --fold ...`
- 增加测试与 CI（GitHub Actions）
- README 提供数据样例格式、输出文件解释、推理示例

