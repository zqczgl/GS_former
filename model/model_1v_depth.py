import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        return x + self.pe[: x.size(0), :]


class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim: int, nhead: int = 4, num_layers: int = 1, dim_feedforward: int = 256):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model=input_dim)
        layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, S, C) -> (S, B, C)
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.permute(1, 0, 2)
        return x


class PhysicalPositionEmbedding(nn.Module):
    """Map normalized physical positions to embedding vectors.

    IMPORTANT: padding positions should be masked out outside this module.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: (B, L)
        emb = self.mlp(pos.unsqueeze(-1))
        return self.norm(emb)


class AttnPool1d(nn.Module):
    """Attention pooling over a sequence to yield a single token.

    x: (B, L, C)
    mask: (B, L) with True for valid positions (optional)
    """

    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.v(torch.tanh(self.proj(x))).squeeze(-1)  # (B, L)

        if mask is not None:
            # Ensure boolean mask
            mask = mask.to(dtype=torch.bool)
            # Large negative for padded positions so softmax -> 0
            scores = scores.masked_fill(~mask, float('-inf'))

        weights = torch.softmax(scores, dim=1)  # (B, L)

        if mask is not None:
            weights = weights.masked_fill(~mask, 0.0)

        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)  # (B, C)
        return pooled, weights


class ResConvBlockLayer(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, kernel_size: int, dropout: float = 0.25):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same')
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(in_channels=hidden, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn(x)
        x = x + residual
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x


class ChromosomeCNN(nn.Module):
    def __init__(self, num_snps: int, input_dim: int, kernel_size: int = 5):
        super().__init__()
        # NOTE: num_snps kept for interface compatibility
        self.res_block1 = ResConvBlockLayer(in_channels=input_dim, hidden=16, out_channels=32, kernel_size=kernel_size)
        self.res_block2 = ResConvBlockLayer(in_channels=32, hidden=32, out_channels=64, kernel_size=kernel_size)
        self.res_block3 = ResConvBlockLayer(in_channels=64, hidden=32, out_channels=16, kernel_size=kernel_size)
        self.kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        return x


class PhenotypePredictor_1v(nn.Module):
    """Original baseline model (kept for compatibility)."""

    def __init__(self, num_chromosomes: int, snp_counts: List[int], input_dim: int):
        super().__init__()
        self.chromosome_cnns = nn.ModuleList([ChromosomeCNN(snp_counts[i], input_dim) for i in range(num_chromosomes)])

        with torch.no_grad():
            sample_input = torch.randn(1, input_dim, snp_counts[0])
            sample_output = self.chromosome_cnns[0](sample_input)
            cnn_output_channels = sample_output.size(1)
            cnn_output_length = sample_output.size(2)

        self.transformer_decoder = TransformerEncoderModel(input_dim=cnn_output_channels)
        self.total_sequence_length = cnn_output_length * cnn_output_channels * num_chromosomes
        self.mlp = nn.Sequential(
            nn.Linear(self.total_sequence_length, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, cnn in enumerate(self.chromosome_cnns):
            chromosome_data = x[:, i, :, :].permute(0, 2, 1)
            output = cnn(chromosome_data)
            outputs.append(output)

        combined_output = torch.cat(outputs, dim=2)
        combined_output = combined_output.permute(0, 2, 1)
        combined_output = self.transformer_decoder(combined_output)
        combined_output = torch.flatten(combined_output, 1)
        combined_output = self.mlp(combined_output)
        return combined_output


class PhenotypePredictor_HCA_Pos(nn.Module):
    """HCA variant with explicit physical position embedding + proper masking.

    Improvements implemented:
    - Padding-safe physical position embedding: padding positions are masked out
      so they remain exact zeros after adding embeddings.
    - Masked attention pooling: padded positions do not participate in pooling.
    - Missing-vs-padding disambiguation: a learnable missing embedding is added
      wherever genotype is missing (but not padded).

    Inputs
    ------
    x: (B, num_chromosomes, L, input_dim)
    missing_mask: (B, num_chromosomes, L) bool, True indicates missing genotype
                  at that locus. Optional but recommended.
    """

    def __init__(
        self,
        num_chromosomes: int,
        snp_counts: List[int],
        input_dim: int,
        phys_pos: torch.Tensor,
        pad_mask: torch.Tensor,
        pos_hidden_dim: int = 32,
        pool_hidden_dim: int = 64,
        cnn_kernel_size: int = 5,
    ):
        super().__init__()
        self.num_chromosomes = int(num_chromosomes)
        self.input_dim = int(input_dim)
        self.cnn_kernel_size = int(cnn_kernel_size)

        self.chromosome_cnns = nn.ModuleList(
            [ChromosomeCNN(snp_counts[i], input_dim, kernel_size=cnn_kernel_size) for i in range(num_chromosomes)]
        )

        # Explicit physical position embedding
        self.pos_encoder = PhysicalPositionEmbedding(embed_dim=input_dim, hidden_dim=pos_hidden_dim, dropout=0.1)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))

        # Missing embedding (distinguish missing genotype vs padding)
        self.missing_emb = nn.Parameter(torch.zeros(input_dim))
        self.missing_scale = nn.Parameter(torch.tensor(1.0))

        # Register buffers (moved with .to(device))
        assert phys_pos is not None, 'phys_pos is required, shape=(num_chromosomes, L)'
        assert pad_mask is not None, 'pad_mask is required, shape=(num_chromosomes, L)'

        self.register_buffer('phys_pos', phys_pos.float())
        self.register_buffer('pad_mask', pad_mask.to(dtype=torch.bool))

        with torch.no_grad():
            sample_input = torch.randn(1, input_dim, snp_counts[0])
            sample_output = self.chromosome_cnns[0](sample_input)
            cnn_output_channels = sample_output.size(1)
            cnn_output_length = sample_output.size(2)

        self.attn_pool = AttnPool1d(dim=cnn_output_channels, hidden_dim=pool_hidden_dim)
        self.chr_transformer = TransformerEncoderModel(input_dim=cnn_output_channels)

        self.total_sequence_length = cnn_output_channels * num_chromosomes
        self.mlp = nn.Sequential(
            nn.Linear(self.total_sequence_length, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

        # cache expected post-CNN length for mask pooling sanity check
        self._cnn_out_length = int(cnn_output_length)

    def _downsample_mask_through_cnn(self, mask: torch.Tensor) -> torch.Tensor:
        """Approximate propagation of a boolean mask through the CNN's MaxPool layers.

        The ChromosomeCNN uses 3 MaxPool1d layers with kernel_size=stride=cnn_kernel_size.
        We downsample the mask with max_pool1d so that any valid SNP in a pooling
        window makes the pooled position valid.
        """
        # mask: (B, L) bool
        m = mask.to(dtype=torch.float32).unsqueeze(1)  # (B,1,L)
        for _ in range(3):
            m = F.max_pool1d(m, kernel_size=self.cnn_kernel_size, stride=self.cnn_kernel_size)
        m = m.squeeze(1)  # (B, L_out)
        # Safety: enforce shape match if possible
        if m.size(1) != self._cnn_out_length:
            # If mismatch occurs due to config changes, fall back to trimming/padding
            if m.size(1) > self._cnn_out_length:
                m = m[:, : self._cnn_out_length]
            else:
                pad = self._cnn_out_length - m.size(1)
                m = F.pad(m, (0, pad), value=0.0)
        return m > 0.5

    def forward(self, x: torch.Tensor, missing_mask: Optional[torch.Tensor] = None, return_attn: bool = False):
        B = x.size(0)
        chrom_tokens: List[torch.Tensor] = []
        pool_attns: List[torch.Tensor] = []

        for i, cnn in enumerate(self.chromosome_cnns):
            chromosome_data = x[:, i, :, :]  # (B, L, D)

            # padding mask for this chromosome (same for all samples)
            valid = self.pad_mask[i].unsqueeze(0).expand(B, -1)  # (B, L) bool

            # Add physical position embedding ONLY to valid positions
            pos = self.phys_pos[i].unsqueeze(0).expand(B, -1)  # (B, L)
            # Replace invalid pos with 0 to keep pos_encoder numerically stable
            pos_safe = torch.where(valid, pos, torch.zeros_like(pos))
            pos_emb = self.pos_encoder(pos_safe)  # (B, L, D)
            pos_emb = pos_emb * valid.unsqueeze(-1).to(pos_emb.dtype)
            chromosome_data = chromosome_data + self.pos_scale * pos_emb

            # Add missing embedding (distinguish missing vs padding)
            if missing_mask is not None:
                miss = missing_mask[:, i, :].to(dtype=torch.bool)
                # Ensure we don't mark padding positions as missing
                miss = miss & valid
                chromosome_data = chromosome_data + self.missing_scale * miss.unsqueeze(-1).to(chromosome_data.dtype) * self.missing_emb

            # CNN expects (B, C, L)
            chromosome_data = chromosome_data.permute(0, 2, 1)
            feat = cnn(chromosome_data)  # (B, Cc, Lc)
            feat_seq = feat.permute(0, 2, 1)  # (B, Lc, Cc)

            # Downsample mask to match CNN output length
            pooled_mask = self._downsample_mask_through_cnn(valid)  # (B, Lc)

            token, attn_w = self.attn_pool(feat_seq, mask=pooled_mask)
            chrom_tokens.append(token)
            pool_attns.append(attn_w)

        H = torch.stack(chrom_tokens, dim=1)  # (B, num_chr, Cc)
        H = self.chr_transformer(H)
        H = H.reshape(B, -1)
        out = self.mlp(H)

        if return_attn:
            return out, pool_attns
        return out


def print_output_shape(module, input, output):
    if isinstance(output, tuple):
        print(f"Module: {module.__class__.__name__}, Output is a tuple with {len(output)} elements")
        for i, item in enumerate(output):
            if hasattr(item, 'shape'):
                print(f"  Element {i}: Shape {item.shape}")
            else:
                print(f"  Element {i}: Not a tensor")
    else:
        print(f"Module: {module.__class__.__name__}, Output Shape: {output.shape}")


def register_hooks(model):
    hooks = []
    for _, module in model.named_modules():
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hook = module.register_forward_hook(print_output_shape)
            hooks.append(hook)
    return hooks


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    B = 2
    num_chromosomes = 10
    L = 1000
    D = 8

    x = torch.randn(B, num_chromosomes, L, D).to(device)
    phys_pos = torch.linspace(0, 1, L).repeat(num_chromosomes, 1)
    pad_mask = torch.ones(num_chromosomes, L, dtype=torch.bool)
    missing_mask = torch.zeros(B, num_chromosomes, L, dtype=torch.bool)

    model = PhenotypePredictor_HCA_Pos(
        num_chromosomes=num_chromosomes,
        snp_counts=[L] * num_chromosomes,
        input_dim=D,
        phys_pos=phys_pos,
        pad_mask=pad_mask,
    ).to(device)

    y = model(x, missing_mask=missing_mask)
    print('Output shape:', y.shape)
