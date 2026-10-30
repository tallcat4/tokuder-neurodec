"""TOKUDER TCH channel code (ARIB STD-T99 Annex 2 ch. 5: 72 bits / 20 ms = 48 voice + 24 FEC).

The layout is not in the standard; it was recovered from received frames alone
(see docs/fec.md): the received 72-bit words span a 48-dimensional linear subspace
over GF(2), and its weight distribution (via MacWilliams) is exactly that of
two extended Golay [24,12,8] codes plus 24 uncoded bits. The bits are interleaved by 3:

  position 3i     -> Golay word A, bit i   (12 voice bits + 12 parity)
  position 3i + 1 -> Golay word B, bit i   (12 voice bits + 12 parity)
  position 3i + 2 -> uncoded voice bit i

Both Golay words use the same code, given here by its parity-check matrix in the received
bit order. Which 12 of the 24 positions carry the voice bits is not known (any information
set gives the same 4096 words); `info48` uses a fixed information set, so the 48 voice bits
it returns are a fixed linear relabelling of the true ones, which a learned decoder does not mind.
"""
import numpy as np

N_BITS = 72
GOLAY_A = np.arange(0, 72, 3)
GOLAY_B = np.arange(1, 72, 3)
UNCODED = np.arange(2, 72, 3)

_H_ROWS = [
    "100000010001011100100100", "010000010001010001001110", "001000010001001000011101",
    "000100010001000101111000", "000010010000011000111010", "000001010000010101100011",
    "000000110000001101010110", "000000001001011001101001", "000000000101010101010101",
    "000000000011001100110011", "000000000000111100001111", "000000000000000011111111",
]
H = np.array([[int(c) for c in r] for r in _H_ROWS], np.uint8)  # [12, 24]
# information set: the columns that are not pivots of H's reduced row echelon form
INFO = np.array([7, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23])
_PIVOTS = np.array([c for c in range(24) if c not in INFO])


def _codebook() -> np.ndarray:
    """All 4096 codewords [4096, 24], row k has info bits = binary k (MSB first) on INFO."""
    k = np.arange(4096)
    info = ((k[:, None] >> np.arange(11, -1, -1)) & 1).astype(np.uint8)
    cw = np.zeros((4096, 24), np.uint8)
    cw[:, INFO] = info
    # H is in RREF with identity on _PIVOTS: pivot bit r = sum of H[r, INFO] * info
    cw[:, _PIVOTS] = (info @ H[:, INFO].T) % 2
    return cw


CODEBOOK = _codebook()
assert not ((CODEBOOK @ H.T) % 2).any()
_W = 1 << np.arange(11, -1, -1)


def _coset_leaders() -> tuple[np.ndarray, np.ndarray]:
    """Syndrome (12-bit int) -> minimum-weight error pattern [4096, 24] and its weight."""
    leader = np.zeros((4096, 24), np.uint8)
    weight = np.full(4096, 99)
    weight[0] = 0
    from itertools import combinations
    for w in range(1, 5):
        for pos in combinations(range(24), w):
            e = np.zeros(24, np.uint8)
            e[list(pos)] = 1
            s = int(((H @ e) % 2) @ _W)
            if weight[s] > w:
                weight[s], leader[s] = w, e
    return leader, weight


LEADER, LEADER_WEIGHT = _coset_leaders()  # weight 4 syndromes are detected, not reliably corrected


def syndrome(bits: np.ndarray) -> np.ndarray:
    """[..., 72] -> [..., 2] 12-bit syndromes of Golay words A and B (0 = valid)."""
    out = np.stack([((bits[..., g] @ H.T) % 2) @ _W for g in (GOLAY_A, GOLAY_B)], axis=-1)
    return out.astype(np.int64)


def hard_decode(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bounded-distance decoding (3 errors per Golay word). Returns corrected bits and the
    number of corrected errors per word [..., 2] (4 = detected but uncorrectable, left as is)."""
    bits = bits.astype(np.uint8).copy()
    s = syndrome(bits)
    nerr = LEADER_WEIGHT[s]
    for k, g in enumerate((GOLAY_A, GOLAY_B)):
        fix = nerr[..., k] <= 3
        e = LEADER[s[..., k]] * fix[..., None]
        bits[..., g] ^= e
    return bits, nerr


def soft_decode(llr: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood decoding of each Golay word from LLRs (positive = 1).
    Returns hard bits [N, 72] with both Golay words replaced by the most likely codeword
    (uncoded bits = sign) and the Hamming distance from the hard decision [N, 2]."""
    llr = np.asarray(llr, np.float32).reshape(-1, N_BITS)
    hard = (llr > 0).astype(np.uint8)
    out, dist = hard.copy(), np.zeros((len(llr), 2), np.int64)
    sign = CODEBOOK.astype(np.float32) * 2 - 1  # [4096, 24]
    for k, g in enumerate((GOLAY_A, GOLAY_B)):
        for a in range(0, len(llr), chunk):
            best = np.argmax(llr[a : a + chunk, g] @ sign.T, axis=1)
            out[a : a + chunk, g] = CODEBOOK[best]
        dist[:, k] = (out[:, g] != hard[:, g]).sum(1)
    return out, dist


def info48(bits: np.ndarray) -> np.ndarray:
    """Corrected 72-bit frames [..., 72] -> 48 voice bits [..., 48]: A info (12), B info (12), uncoded (24)."""
    return np.concatenate([bits[..., GOLAY_A[INFO]], bits[..., GOLAY_B[INFO]], bits[..., UNCODED]], axis=-1)


def encode48(info: np.ndarray) -> np.ndarray:
    """Inverse of info48 for valid codewords: 48 voice bits -> 72-bit TCH frame."""
    info = np.asarray(info, np.uint8)
    out = np.zeros(info.shape[:-1] + (N_BITS,), np.uint8)
    for k, g in enumerate((GOLAY_A, GOLAY_B)):
        idx = info[..., 12 * k : 12 * k + 12] @ _W
        out[..., g] = CODEBOOK[idx]
    out[..., UNCODED] = info[..., 24:]
    return out
