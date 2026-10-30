import numpy as np

from tokuder_neurodec import fec as F


def test_arib_test_vector_is_a_codeword():
    # ARIB STD-T99 Annex 2 Table 7-1 (1031 Hz tone); not used to derive the code
    tv = "101011111011011011101111101101011001110000100000100010100000011011111101"
    assert (F.syndrome(np.array([int(c) for c in tv], np.uint8)) == 0).all()


def test_golay_parameters():
    w = F.CODEBOOK.sum(1)
    assert len(np.unique(F.CODEBOOK, axis=0)) == 4096 and w[1:].min() == 8
    assert sorted(set(w.tolist())) == [0, 8, 12, 16, 24]


def test_roundtrip_and_correction():
    rng = np.random.default_rng(0)
    info = rng.integers(0, 2, (2000, 48)).astype(np.uint8)
    cw = F.encode48(info)
    assert (F.syndrome(cw) == 0).all() and (F.info48(cw) == info).all()
    err = cw.copy()
    for row in err:  # 3 errors in each Golay word
        row[rng.choice(F.GOLAY_A, 3, replace=False)] ^= 1
        row[rng.choice(F.GOLAY_B, 3, replace=False)] ^= 1
    fixed, n = F.hard_decode(err)
    assert (fixed == cw).all() and (n == 3).all()
    llr = (cw.astype(np.float32) * 2 - 1) * 40
    llr[:, F.GOLAY_A[:5]] *= -0.1  # 5 weak wrong bits: beyond hard decoding, within soft ML
    soft, _ = F.soft_decode(llr)
    assert (soft[:, F.GOLAY_A] == cw[:, F.GOLAY_A]).mean() > 0.99
