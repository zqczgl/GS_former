import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression


def genotype_to_dataframe(genotype_series: pd.Series) -> pd.DataFrame:
    """Split genotype strings into a token-per-SNP dataframe.

    Each row in `genotype_series` is expected to be a whitespace-separated
    string like: "AA AT 00 ..." where each token corresponds to one SNP.
    """
    genotype_list = genotype_series.astype(str).str.split(" ")
    return pd.DataFrame(genotype_list.tolist())


def _normalize_token_array(seq_all_df: pd.DataFrame) -> np.ndarray:
    """Normalize genotype tokens to uppercase 2-char strings, NaN -> '00'."""
    tokens = np.asarray(seq_all_df.fillna("00").astype(str).to_numpy(), dtype="U")
    tokens = np.char.upper(tokens)
    # ensure exactly 2 chars (pad with '0' on the right)
    tokens = np.char.ljust(tokens, 2, fillchar="0").astype("<U2")
    return tokens


def base_one_hot_encoding_1v(seq_all_df: pd.DataFrame) -> pd.DataFrame:
    """Encode genotype into 1D integer codes per SNP.

    Supports two input formats:
      1) Token-per-SNP: each cell like 'AA', 'AT', '00', 'A0', etc.
      2) Allele-pairs: each SNP represented by two adjacent columns with single
         alleles like 'A' 'T'.

    Unknown / missing genotypes map to -1.
    """
    trans_code = {
        "AA": 0,
        "AT": 1,
        "TA": 1,
        "AC": 2,
        "CA": 2,
        "AG": 3,
        "GA": 3,
        "TT": 4,
        "TC": 5,
        "CT": 5,
        "TG": 6,
        "GT": 6,
        "CC": 7,
        "CG": 8,
        "GC": 8,
        "GG": 9,
        # Missing / partial
        "00": -1,
        "A0": -1,
        "0A": -1,
        "T0": -1,
        "0T": -1,
        "C0": -1,
        "0C": -1,
        "G0": -1,
        "0G": -1,
    }

    tokens = _normalize_token_array(seq_all_df)  # (n, m) tokens of len==2

    n_rows, n_cols = tokens.shape

    # Heuristic: allele-pair format if the first token looks like a single allele
    # and total columns are even.
    first = tokens[0, 0]
    is_allele_pair = (n_cols % 2 == 0) and (isinstance(first, str) and len(first.strip()) == 1)

    if is_allele_pair:
        # If it's allele-pair format, tokens were padded to len==2 (e.g. 'A0'),
        # so we need to re-read raw single-allele columns.
        raw = np.asarray(seq_all_df.fillna("0").astype(str).to_numpy(), dtype="U")
        raw = np.char.upper(raw)
        out_cols = n_cols // 2
        code_arr = np.empty((n_rows, out_cols), dtype=np.int16)
        for i in range(0, n_cols, 2):
            joint = np.char.add(raw[:, i], raw[:, i + 1])
            joint = np.char.ljust(joint, 2, fillchar="0").astype("<U2")
            code_arr[:, i // 2] = np.vectorize(lambda z: trans_code.get(z, -1), otypes=[np.int16])(joint)
        return pd.DataFrame(code_arr)

    # Token-per-SNP format
    # Vectorized mapping via pandas factorization + dict lookup (fast enough & robust)
    flat = tokens.reshape(-1)
    uniq, inv = np.unique(flat, return_inverse=True)
    mapped = np.array([trans_code.get(u, -1) for u in uniq], dtype=np.int16)
    code_arr = mapped[inv].reshape(n_rows, n_cols)
    return pd.DataFrame(code_arr)


def base_one_hot_encoding_8v_dif(seq_all_df: pd.DataFrame) -> pd.DataFrame:
    """8D allele-order-preserving encoding.

    Each SNP token is a 2-char string (a1a2). Each allele maps to a 4D one-hot:
    A,C,G,T -> one-hot; missing/unknown -> all-zeros. Two alleles are
    concatenated -> 8D per SNP.

    NOTE: Missing genotype '00' yields an all-zero 8D vector. Downstream code
    should use an explicit missing-mask to distinguish missing vs padding.
    """
    tokens = _normalize_token_array(seq_all_df)

    n_samples, n_snps = tokens.shape

    # Split into two allele characters (vectorized via view)
    chars = tokens.view("<U1").reshape(n_samples, n_snps, 2)
    a1 = chars[:, :, 0]
    a2 = chars[:, :, 1]

    # Map characters to indices: A=0, C=1, G=2, T=3, 0/unknown=4
    lut = np.full(256, 4, dtype=np.int16)
    lut[ord("A")] = 0
    lut[ord("C")] = 1
    lut[ord("G")] = 2
    lut[ord("T")] = 3
    lut[ord("0")] = 4

    a1_u8 = a1.astype("S1").view(np.uint8)
    a2_u8 = a2.astype("S1").view(np.uint8)
    idx1 = lut[a1_u8]
    idx2 = lut[a2_u8]

    # 4-dim one-hot per allele, row 4 = all zeros (missing)
    base4 = np.zeros((5, 4), dtype=np.int8)
    base4[0, 0] = 1  # A
    base4[1, 1] = 1  # C
    base4[2, 2] = 1  # G
    base4[3, 3] = 1  # T

    one_hot = np.concatenate([base4[idx1], base4[idx2]], axis=2)  # (n, snps, 8)
    return pd.DataFrame(one_hot.reshape(n_samples, -1))


def sanitize_discrete_codes(X: Union[np.ndarray, pd.DataFrame], missing_code: int = -1) -> np.ndarray:
    """Map missing_code to a new non-negative category for MI estimation.

    mutual_info_regression(discrete_features=True) treats values as discrete
    categories. Using a dedicated code for missing improves stability.
    """
    Xn = np.asarray(X)
    if Xn.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={Xn.shape}")

    # Determine max code ignoring missing
    valid = Xn != missing_code
    if np.any(valid):
        max_code = int(np.max(Xn[valid]))
    else:
        max_code = 0
    miss_repl = max_code + 1
    Xc = Xn.copy()
    Xc[~valid] = miss_repl
    return Xc


class MISelector(BaseEstimator, TransformerMixin):
    """Select top-k SNP loci by association score.

    Default method is *mutual information* (MI) via sklearn's
    mutual_info_regression with discrete features.

    If method='mic', will try to compute MIC via minepy (optional dependency).
    When minepy is not installed, it falls back to MI with a warning.
    """

    def __init__(
        self,
        k: int = 1000,
        method: str = "mi",
        n_neighbors: int = 7,
        random_state: Optional[int] = 42,
        missing_code: int = -1,
    ):
        self.k = int(k)
        self.method = str(method).lower()
        self.n_neighbors = int(n_neighbors)
        self.random_state = random_state
        self.missing_code = int(missing_code)

        self.scores_: Optional[np.ndarray] = None
        self.top_k_indices_: Optional[np.ndarray] = None

    def fit(self, X, y):
        Xn = sanitize_discrete_codes(X, missing_code=self.missing_code)
        y = np.asarray(y).reshape(-1)

        if self.method == "mic":
            try:
                from minepy import MINE  # type: ignore

                scores = np.zeros(Xn.shape[1], dtype=np.float32)
                mine = MINE(alpha=0.6, c=15)
                for j in range(Xn.shape[1]):
                    mine.compute_score(Xn[:, j], y)
                    scores[j] = mine.mic()
                self.scores_ = scores
            except Exception as e:
                warnings.warn(
                    f"method='mic' requested but minepy is not available ({e}). "
                    f"Falling back to MI (mutual_info_regression).",
                    RuntimeWarning,
                )
                self.scores_ = mutual_info_regression(
                    Xn,
                    y,
                    discrete_features=True,
                    n_neighbors=self.n_neighbors,
                    random_state=self.random_state,
                )
        else:
            # default: MI
            self.scores_ = mutual_info_regression(
                Xn,
                y,
                discrete_features=True,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )

        k_eff = min(self.k, self.scores_.shape[0])
        # argsort ascending; take top-k
        self.top_k_indices_ = np.argsort(self.scores_)[-k_eff:]
        return self

    def transform(self, X):
        if self.top_k_indices_ is None:
            raise RuntimeError("MISelector must be fit before transform")
        Xn = np.asarray(X)
        return Xn[:, self.top_k_indices_]


# Backward compatibility: old name used in some scripts.
class MICSelector(MISelector):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "MICSelector is deprecated. Use MISelector(method='mi' or 'mic') instead. "
            "This project defaults to MI unless minepy is installed for MIC.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)
