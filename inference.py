"""
推理脚本 - 使用训练好的模型预测新样本的表型

使用方法:
    python inference.py \
        --model output/saved_models/model_fold_1.pth \
        --metadata output/saved_models/metadata_fold_1.json \
        --input data/new_samples.csv \
        --map data/dpc_data_100SW.map \
        --output predictions.csv
"""
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from data_process.encode import (
    genotype_to_dataframe,
    base_one_hot_encoding_8v_dif,
    base_one_hot_encoding_1v,
)
from model.model_1v_depth import PhenotypePredictor_HCA_Pos


@dataclass
class InferenceConfig:
    """推理配置"""
    model_path: str           # 模型权重路径 (.pth)
    metadata_path: str        # 元数据路径 (.json)
    input_csv: str            # 输入基因型CSV
    map_path: str             # SNP位置MAP文件
    output_csv: str = 'predictions.csv'
    batch_size: int = 64
    device: str = 'auto'


def load_map(map_path: str) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    """加载MAP文件，返回DataFrame和染色体范围列表"""
    map_df = pd.read_csv(
        map_path,
        sep='\t',
        header=None,
        names=['chromosome', 'snp_id', 'map', 'position'],
    )
    ranges: List[Tuple[int, int]] = []
    start = 0
    for _, grp in map_df.groupby('chromosome', sort=False):
        end = start + len(grp)
        ranges.append((start, end))
        start = end
    return map_df, ranges


def load_model_with_metadata(cfg: InferenceConfig) -> Tuple[torch.nn.Module, dict]:
    """加载模型和推理所需的元数据
    
    Returns:
        model: 加载好权重的模型
        metadata: 包含scaler、device等信息的字典
    """
    # 加载元数据
    with open(cfg.metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 确定设备
    if cfg.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = cfg.device
    
    # 重建模型结构
    phys_pos = torch.tensor(metadata['phys_pos'], dtype=torch.float32)
    pad_mask = torch.tensor(metadata['pad_mask'], dtype=torch.bool)
    
    model = PhenotypePredictor_HCA_Pos(
        num_chromosomes=metadata['num_chromosomes'],
        snp_counts=[metadata['selected_snp_count']] * metadata['num_chromosomes'],
        input_dim=8,
        phys_pos=phys_pos,
        pad_mask=pad_mask,
    )
    
    # 加载权重
    state_dict = torch.load(cfg.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 重建StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(metadata['scaler_mean'])
    scaler.scale_ = np.array(metadata['scaler_scale'])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = 1
    
    metadata['scaler'] = scaler
    metadata['device'] = device
    
    return model, metadata


def preprocess_for_inference(
    input_csv: str,
    map_path: str,
    metadata: dict,
) -> Tuple[torch.Tensor, torch.Tensor, List]:
    """预处理推理数据
    
    使用训练时保存的SNP选择索引处理新数据
    
    Returns:
        X: 基因型张量 (N, num_chr, k, 8)
        missing_mask: 缺失掩码 (N, num_chr, k)
        sample_ids: 样本ID列表
    """
    map_df, chrom_ranges = load_map(map_path)
    n_snps = len(map_df)
    
    # 读取输入数据
    df = pd.read_csv(input_csv, sep='\t')
    
    # 获取样本ID
    if 'sample_id' in df.columns:
        sample_ids = df['sample_id'].tolist()
    else:
        sample_ids = list(range(len(df)))
    
    # 解析基因型
    genotype_df = genotype_to_dataframe(df['genotype'])
    if genotype_df.shape[1] != n_snps:
        raise ValueError(
            f'SNP count mismatch: CSV has {genotype_df.shape[1]} SNPs, '
            f'MAP has {n_snps} SNPs'
        )
    
    # 编码
    one_hot_8v = base_one_hot_encoding_8v_dif(genotype_df).to_numpy(dtype=np.float32)
    geno_1v = base_one_hot_encoding_1v(genotype_df).to_numpy(dtype=np.int16)
    
    n_samples = len(df)
    one_hot_8v = one_hot_8v.reshape(n_samples, n_snps, 8)
    
    # 使用训练时保存的索引选择SNP
    selected_rel = [np.array(idx, dtype=np.int64) for idx in metadata['selected_rel_indices']]
    k = metadata['selected_snp_count']
    
    chrom_data_list: List[np.ndarray] = []
    missing_list: List[np.ndarray] = []
    
    for (start, end), idx in zip(chrom_ranges, selected_rel):
        # 提取该染色体的数据
        chrom_8v = one_hot_8v[:, start:end, :]
        selected = chrom_8v[:, idx, :]
        
        # Padding
        real_len = selected.shape[1]
        padded = np.zeros((n_samples, k, 8), dtype=np.float32)
        padded[:, :real_len, :] = selected
        
        # 计算缺失掩码
        chrom_1v = geno_1v[:, start:end]
        sel_1v = chrom_1v[:, idx]
        miss = (sel_1v == -1)
        
        miss_padded = np.zeros((n_samples, k), dtype=bool)
        miss_padded[:, :real_len] = miss
        
        chrom_data_list.append(padded)
        missing_list.append(miss_padded)
    
    X = torch.tensor(np.stack(chrom_data_list, axis=1), dtype=torch.float32)
    missing_mask = torch.tensor(np.stack(missing_list, axis=1), dtype=torch.bool)
    
    return X, missing_mask, sample_ids


def predict_batch(
    model: torch.nn.Module,
    X: torch.Tensor,
    missing_mask: torch.Tensor,
    scaler: StandardScaler,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """批量预测
    
    Args:
        model: 加载好的模型
        X: 输入张量
        missing_mask: 缺失掩码
        scaler: 用于反标准化的scaler
        device: 计算设备
        batch_size: 批量大小
    
    Returns:
        predictions: 反标准化后的预测值数组
    """
    model.eval()
    predictions = []
    
    n_samples = X.size(0)
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size].to(device)
            batch_mask = missing_mask[i:i+batch_size].to(device)
            
            out = model(batch_X, missing_mask=batch_mask)
            predictions.append(out.cpu().numpy())
    
    # 合并所有batch的结果
    preds_scaled = np.concatenate(predictions, axis=0).flatten()
    
    # 反标准化，恢复到原始表型尺度
    preds_original = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    
    return preds_original


def predict_single(
    model: torch.nn.Module,
    genotype_str: str,
    map_path: str,
    metadata: dict,
) -> float:
    """单样本预测
    
    Args:
        model: 加载好的模型
        genotype_str: 空格分隔的基因型字符串
        map_path: MAP文件路径
        metadata: 元数据字典
    
    Returns:
        预测的表型值（反标准化后）
    """
    # 创建临时DataFrame
    df = pd.DataFrame({'genotype': [genotype_str]})
    
    map_df, chrom_ranges = load_map(map_path)
    n_snps = len(map_df)
    
    genotype_df = genotype_to_dataframe(df['genotype'])
    if genotype_df.shape[1] != n_snps:
        raise ValueError(f'SNP count mismatch')
    
    one_hot_8v = base_one_hot_encoding_8v_dif(genotype_df).to_numpy(dtype=np.float32)
    geno_1v = base_one_hot_encoding_1v(genotype_df).to_numpy(dtype=np.int16)
    one_hot_8v = one_hot_8v.reshape(1, n_snps, 8)
    
    selected_rel = [np.array(idx, dtype=np.int64) for idx in metadata['selected_rel_indices']]
    k = metadata['selected_snp_count']
    
    chrom_data_list = []
    missing_list = []
    
    for (start, end), idx in zip(chrom_ranges, selected_rel):
        chrom_8v = one_hot_8v[:, start:end, :]
        selected = chrom_8v[:, idx, :]
        
        real_len = selected.shape[1]
        padded = np.zeros((1, k, 8), dtype=np.float32)
        padded[:, :real_len, :] = selected
        
        chrom_1v = geno_1v[:, start:end]
        sel_1v = chrom_1v[:, idx]
        miss = (sel_1v == -1)
        
        miss_padded = np.zeros((1, k), dtype=bool)
        miss_padded[:, :real_len] = miss
        
        chrom_data_list.append(padded)
        missing_list.append(miss_padded)
    
    X = torch.tensor(np.stack(chrom_data_list, axis=1), dtype=torch.float32)
    missing_mask = torch.tensor(np.stack(missing_list, axis=1), dtype=torch.bool)
    
    device = metadata['device']
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        missing_mask = missing_mask.to(device)
        out = model(X, missing_mask=missing_mask)
    
    pred_scaled = out.cpu().numpy().flatten()[0]
    pred_original = metadata['scaler'].inverse_transform([[pred_scaled]])[0, 0]
    
    return float(pred_original)


def main():
    parser = argparse.ArgumentParser(
        description='使用训练好的模型进行表型预测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python inference.py --model output/saved_models/model_fold_1.pth \\
                      --metadata output/saved_models/metadata_fold_1.json \\
                      --input data/new_samples.csv \\
                      --map data/dpc_data_100SW.map \\
                      --output predictions.csv
        '''
    )
    parser.add_argument('--model', required=True, help='模型权重路径 (.pth)')
    parser.add_argument('--metadata', required=True, help='元数据路径 (.json)')
    parser.add_argument('--input', required=True, help='输入基因型CSV (TAB分隔)')
    parser.add_argument('--map', required=True, help='SNP位置MAP文件')
    parser.add_argument('--output', default='predictions.csv', help='输出CSV路径 (默认: predictions.csv)')
    parser.add_argument('--batch-size', type=int, default=64, help='批量大小 (默认: 64)')
    parser.add_argument('--device', default='auto', help='计算设备 (auto/cpu/cuda, 默认: auto)')
    
    args = parser.parse_args()
    
    cfg = InferenceConfig(
        model_path=args.model,
        metadata_path=args.metadata,
        input_csv=args.input,
        map_path=args.map,
        output_csv=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # 验证文件存在
    for path, name in [(cfg.model_path, '模型'), (cfg.metadata_path, '元数据'), 
                       (cfg.input_csv, '输入CSV'), (cfg.map_path, 'MAP文件')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'{name}文件不存在: {path}')
    
    print(f'正在加载模型: {cfg.model_path}')
    model, metadata = load_model_with_metadata(cfg)
    print(f'模型已加载到设备: {metadata["device"]}')
    print(f'  - 染色体数量: {metadata["num_chromosomes"]}')
    print(f'  - 每条染色体SNP数: {metadata["selected_snp_count"]}')
    
    print(f'\n正在处理输入数据: {cfg.input_csv}')
    X, missing_mask, sample_ids = preprocess_for_inference(cfg.input_csv, cfg.map_path, metadata)
    print(f'  - 样本数量: {X.size(0)}')
    print(f'  - 数据形状: {X.shape}')
    
    print('\n正在执行预测...')
    predictions = predict_batch(
        model, X, missing_mask,
        metadata['scaler'], metadata['device'],
        cfg.batch_size
    )
    
    # 保存结果
    result_df = pd.DataFrame({
        'sample_id': sample_ids,
        'predicted_phenotype': predictions
    })
    result_df.to_csv(cfg.output_csv, index=False)
    
    print(f'\n预测完成!')
    print(f'  - 结果已保存到: {cfg.output_csv}')
    print(f'  - 预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]')
    print(f'  - 预测值均值: {predictions.mean():.4f}')


if __name__ == '__main__':
    main()
