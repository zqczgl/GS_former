import json
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr

from data_process.encode import (
    genotype_to_dataframe,
    base_one_hot_encoding_8v_dif,
    base_one_hot_encoding_1v,
    MISelector,
)
from model.model_1v_depth import PhenotypePredictor_HCA_Pos
from model.early_stop import EarlyStopping


@dataclass
class TrainConfig:
    csv_path: str = 'data/dpc_data_100SW.csv'
    map_path: str = 'data/dpc_data_100SW.map'

    chunksize: int = 100
    selected_snp_count: int = 1000

    test_ratio: float = 0.1
    n_splits: int = 10

    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3

    selector_method: str = 'mi'  # 'mi' (default) or 'mic' (requires minepy)

    seed: int = 42

    out_dir: str = 'output'


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """读取 MAP 文件，返回 1.完整的数据表 2.染色体索引范围 3.物理位置数组列表"""
def load_map(map_path: str) -> Tuple[pd.DataFrame, List[Tuple[int, int]], List[np.ndarray]]:
    map_df = pd.read_csv(
        map_path,
        sep='\t',
        header=None,
        names=['chromosome', 'snp_id', 'map', 'position'],
    )

    ranges: List[Tuple[int, int]] = []
    positions: List[np.ndarray] = []

    start = 0
    for _, grp in map_df.groupby('chromosome', sort=False):
        count = len(grp)
        end = start + count
        ranges.append((start, end))
        positions.append(grp['position'].values.astype(np.float32))
        start = end

    return map_df, ranges, positions


def read_and_encode_all(cfg: TrainConfig, n_snps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read CSV in chunks and encode genotypes into 8v and 1v arrays.

    Returns
    -------
    one_hot_8v: (N, n_snps, 8) float32
    geno_1v: (N, n_snps) int16
    phenotype_raw: (N,) float32
    """
    print('Starting chunked data reading...')
    merged_df_chunks = pd.read_csv(cfg.csv_path, sep='\t', chunksize=cfg.chunksize)

    one_hot_8v_list = []
    geno_1v_list = []
    phenotype_list = []

    for chunk_id, chunk in enumerate(merged_df_chunks, start=1):
        print(f'Processing chunk {chunk_id}...')

        genotype_data = chunk['genotype']
        phenotype_data = chunk['phenotype'].values.astype(np.float32)

        genotype_df = genotype_to_dataframe(genotype_data)
        if genotype_df.shape[1] != n_snps:
            raise ValueError(
                f'Genotype token count mismatch: got {genotype_df.shape[1]} SNPs in CSV but MAP has {n_snps}. '
                f'Please ensure CSV genotype columns align with MAP.'
            )

        chunk_one_hot_8v = base_one_hot_encoding_8v_dif(genotype_df).to_numpy(dtype=np.float32)
        chunk_geno_1v = base_one_hot_encoding_1v(genotype_df).to_numpy(dtype=np.int16)

        one_hot_8v_list.append(chunk_one_hot_8v)
        geno_1v_list.append(chunk_geno_1v)
        phenotype_list.append(phenotype_data)

    print('Merging all data chunks...')
    one_hot_8v_flat = np.concatenate(one_hot_8v_list, axis=0)
    geno_1v = np.concatenate(geno_1v_list, axis=0)
    phenotype_raw = np.concatenate(phenotype_list, axis=0)

    n_samples = phenotype_raw.shape[0]
    one_hot_8v = one_hot_8v_flat.reshape(n_samples, n_snps, 8).astype(np.float32, copy=False)

    return one_hot_8v, geno_1v, phenotype_raw


def select_snps_per_chromosome(
    geno_1v: np.ndarray,
    y_train: np.ndarray,
    train_ids: np.ndarray,
    chrom_ranges: List[Tuple[int, int]],
    chrom_positions: List[np.ndarray],
    k: int,
    selector_method: str,
) -> Tuple[List[np.ndarray], torch.Tensor, torch.Tensor]:
    """Fit selector on TRAIN ONLY and return selected indices + (phys_pos, pad_mask) buffers.

    Returns
    -------
    selected_rel_indices: list of arrays (len=num_chr). Indices are relative to each chromosome slice.
    phys_pos_tensor: (num_chr, k) float32, padded tail = -1
    pad_mask_tensor: (num_chr, k) bool, True for real SNPs
    """
    selected_rel: List[np.ndarray] = []
    phys_pos_list: List[np.ndarray] = []
    pad_mask_list: List[np.ndarray] = []

    for (start, end), pos in zip(chrom_ranges, chrom_positions):
        X_chr = geno_1v[train_ids, start:end]

        selector = MISelector(k=k, method=selector_method, missing_code=-1, random_state=42)
        selector.fit(X_chr, y_train)
        idx = selector.top_k_indices_  # relative indices

        # re-sort by physical position
        pos_sel = pos[idx]
        order = np.argsort(pos_sel)
        idx = idx[order]
        pos_sel = pos_sel[order]

        # normalize pos to [0,1] for this chromosome
        pos_min = float(np.min(pos))
        pos_max = float(np.max(pos))
        denom = (pos_max - pos_min) if (pos_max - pos_min) > 0 else 1.0
        pos_norm = (pos_sel - pos_min) / (denom + 1e-8)

        real_len = int(len(idx))
        pad_mask = np.zeros((k,), dtype=bool)
        pad_mask[:real_len] = True

        pos_vec = np.full((k,), -1.0, dtype=np.float32)
        pos_vec[:real_len] = pos_norm.astype(np.float32)

        selected_rel.append(idx.astype(np.int64))
        pad_mask_list.append(pad_mask)
        phys_pos_list.append(pos_vec)

    phys_pos_tensor = torch.tensor(np.stack(phys_pos_list, axis=0), dtype=torch.float32)
    pad_mask_tensor = torch.tensor(np.stack(pad_mask_list, axis=0), dtype=torch.bool)
    return selected_rel, phys_pos_tensor, pad_mask_tensor


def build_split_tensors(
    one_hot_8v: np.ndarray,
    geno_1v: np.ndarray,
    sample_ids: np.ndarray,
    chrom_ranges: List[Tuple[int, int]],
    selected_rel_indices: List[np.ndarray],
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (X, missing_mask) for a split using preselected indices."""
    chrom_data_list: List[np.ndarray] = []
    missing_list: List[np.ndarray] = []

    for (start, end), idx in zip(chrom_ranges, selected_rel_indices):
        # (N, count, 8)
        chrom_8v = one_hot_8v[sample_ids, start:end, :]
        selected = chrom_8v[:, idx, :]

        real_len = selected.shape[1]
        padded = np.zeros((selected.shape[0], k, 8), dtype=np.float32)
        padded[:, :real_len, :] = selected.astype(np.float32, copy=False)

        # Missing mask from 1v codes (-1 indicates missing)
        chrom_1v = geno_1v[sample_ids, start:end]
        sel_1v = chrom_1v[:, idx]
        miss = (sel_1v == -1)

        miss_padded = np.zeros((selected.shape[0], k), dtype=bool)
        miss_padded[:, :real_len] = miss

        chrom_data_list.append(padded)
        missing_list.append(miss_padded)

    X = torch.tensor(np.stack(chrom_data_list, axis=1), dtype=torch.float32)  # (N, num_chr, k, 8)
    missing_mask = torch.tensor(np.stack(missing_list, axis=1), dtype=torch.bool)  # (N, num_chr, k)
    return X, missing_mask


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    """Return (mse, pearson, spearman)"""
    model.eval()
    criterion = torch.nn.MSELoss(reduction='sum')

    preds = []
    targets = []
    loss_sum = 0.0

    with torch.no_grad():
        for x, miss, y in loader:
            x = x.to(device)
            miss = miss.to(device)
            y = y.to(device)

            out = model(x, missing_mask=miss)
            loss_sum += float(criterion(out, y).item())
            preds.append(out.detach().cpu())
            targets.append(y.detach().cpu())

    preds_np = torch.cat(preds).numpy().flatten()
    targets_np = torch.cat(targets).numpy().flatten()
    mse = loss_sum / len(loader.dataset)

    pear = float(pearsonr(preds_np, targets_np)[0])
    spear = float(spearmanr(preds_np, targets_np)[0])
    return mse, pear, spear


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.makedirs(os.path.join(cfg.out_dir, 'saved_models'), exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, 'loss_png'), exist_ok=True)

    # MAP file defines SNP order and chromosome boundaries
    map_df, chrom_ranges, chrom_positions = load_map(cfg.map_path)
    n_snps = len(map_df)
    num_chromosomes = len(chrom_ranges)

    one_hot_8v, geno_1v, phenotype_raw = read_and_encode_all(cfg, n_snps=n_snps)

    n_samples = phenotype_raw.shape[0]
    rng = np.random.default_rng(cfg.seed)
    all_indices = rng.permutation(n_samples)
    test_size = int(cfg.test_ratio * n_samples)
    test_ids = all_indices[:test_size]
    train_val_ids = all_indices[test_size:]

    print(f'Total samples: {n_samples}, train/val: {len(train_val_ids)}, test: {len(test_ids)}')

    kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    cv_results = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_val_ids), start=1):
        print(f'\n=== Fold {fold}/{cfg.n_splits} ===')

        train_ids = train_val_ids[tr_idx]
        val_ids = train_val_ids[va_idx]

        # -------------------------
        # Fit scaler on TRAIN ONLY
        # -------------------------
        scaler = StandardScaler()
        y_train = scaler.fit_transform(phenotype_raw[train_ids].reshape(-1, 1)).flatten().astype(np.float32)
        y_val = scaler.transform(phenotype_raw[val_ids].reshape(-1, 1)).flatten().astype(np.float32)
        y_test = scaler.transform(phenotype_raw[test_ids].reshape(-1, 1)).flatten().astype(np.float32)

        # -------------------------
        # Feature selection on TRAIN ONLY
        # -------------------------
        selected_rel, phys_pos_tensor, pad_mask_tensor = select_snps_per_chromosome(
            geno_1v=geno_1v,
            y_train=y_train,
            train_ids=train_ids,
            chrom_ranges=chrom_ranges,
            chrom_positions=chrom_positions,
            k=cfg.selected_snp_count,
            selector_method=cfg.selector_method,
        )

        # Build split tensors using the same selected indices
        X_train, miss_train = build_split_tensors(one_hot_8v, geno_1v, train_ids, chrom_ranges, selected_rel, cfg.selected_snp_count)
        X_val, miss_val = build_split_tensors(one_hot_8v, geno_1v, val_ids, chrom_ranges, selected_rel, cfg.selected_snp_count)
        X_test, miss_test = build_split_tensors(one_hot_8v, geno_1v, test_ids, chrom_ranges, selected_rel, cfg.selected_snp_count)

        y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

        train_ds = TensorDataset(X_train, miss_train, y_train_t)
        val_ds = TensorDataset(X_val, miss_val, y_val_t)
        test_ds = TensorDataset(X_test, miss_test, y_test_t)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

        # Model per fold because phys_pos/pad_mask depend on selected SNPs
        model = PhenotypePredictor_HCA_Pos(
            num_chromosomes=num_chromosomes,
            snp_counts=[cfg.selected_snp_count] * num_chromosomes,
            input_dim=8,
            phys_pos=phys_pos_tensor,
            pad_mask=pad_mask_tensor,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            min_lr=1e-6,
        )
        criterion = torch.nn.MSELoss()

        ckpt_path = os.path.join(cfg.out_dir, 'saved_models', f'checkpoint_fold_{fold}.pt')
        early_stop = EarlyStopping(patience=20, delta=0.0, path=ckpt_path, verbose=False)

        best_val = float('inf')

        for epoch in range(cfg.epochs):
            model.train()
            train_loss_sum = 0.0

            for x, miss, y in train_loader:
                x = x.to(device)
                miss = miss.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                out = model(x, missing_mask=miss)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                train_loss_sum += float(loss.item()) * x.size(0)

            train_loss = train_loss_sum / len(train_ds)

            # Validation
            val_mse, val_pear, val_spear = evaluate(model, val_loader, device)
            scheduler.step(val_mse)

            best_val = min(best_val, val_mse)

            lr_now = optimizer.param_groups[0]['lr']
            print(
                f'Epoch {epoch + 1:03d}: '
                f'TrainMSE={train_loss:.6f}  ValMSE={val_mse:.6f}  '
                f'ValPearson={val_pear:.4f}  ValSpearman={val_spear:.4f}  LR={lr_now:.2e}'
            )

            early_stop(val_mse, model, epoch=epoch)
            if early_stop.early_stop:
                print('Early stopping triggered')
                break

        # Load best checkpoint and evaluate on test
        early_stop.load_best(model, map_location=device)

        test_mse, test_pear, test_spear = evaluate(model, test_loader, device)

        cv_results.append(
            {
                'fold': fold,
                'best_val_mse': best_val,
                'test_mse': test_mse,
                'test_pearson': test_pear,
                'test_spearman': test_spear,
                'checkpoint': ckpt_path,
            }
        )

        # Save fold model state_dict
        torch.save(model.state_dict(), os.path.join(cfg.out_dir, 'saved_models', f'model_fold_{fold}.pth'))

        # Save inference metadata (for inference.py)
        inference_metadata = {
            'selected_rel_indices': [idx.tolist() for idx in selected_rel],
            'phys_pos': phys_pos_tensor.cpu().numpy().tolist(),
            'pad_mask': pad_mask_tensor.cpu().numpy().tolist(),
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'chrom_ranges': chrom_ranges,
            'num_chromosomes': num_chromosomes,
            'selected_snp_count': cfg.selected_snp_count,
        }
        metadata_path = os.path.join(cfg.out_dir, 'saved_models', f'metadata_fold_{fold}.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(inference_metadata, f, ensure_ascii=False, indent=2)
        print(f'Inference metadata saved to {metadata_path}')

        print(
            f'Fold {fold} done: TestMSE={test_mse:.6f}, Pearson={test_pear:.4f}, Spearman={test_spear:.4f}'
        )

    print('\n=== Cross-Validation Results ===')
    for res in cv_results:
        print(
            f"Fold {res['fold']}: TestMSE={res['test_mse']:.6f}, Pearson={res['test_pearson']:.4f}, Spearman={res['test_spearman']:.4f}"
        )

    avg_pearson = float(np.mean([r['test_pearson'] for r in cv_results]))
    avg_spearman = float(np.mean([r['test_spearman'] for r in cv_results]))
    print(f'\nAverage Pearson Correlation: {avg_pearson:.4f}')
    print(f'Average Spearman Correlation: {avg_spearman:.4f}')


if __name__ == '__main__':
    main()
