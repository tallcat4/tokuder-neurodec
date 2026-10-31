from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from tokuder_neurodec import cli, decode, fec, frames, load

MODELS = Path(__file__).resolve().parent.parent / "models"
MODEL = MODELS / "tokuder48_causal_v1.pt"  # the default model
LARGE = MODELS / "tokuder48_v1.pt"
TV = "101011111011011011101111101101011001110000100000100010100000011011111101"  # ARIB STD-T99 Table Annex 2-7-1


def random_frames(n, seed=0):
    rng = np.random.default_rng(seed)
    return fec.encode48(rng.integers(0, 2, (n, 48)).astype(np.uint8))


@pytest.mark.parametrize("model", [MODEL, LARGE])
def test_decode_shape_and_range(model):
    G = load(model)
    y = decode(G, fec.info48(random_frames(37)))
    assert y.shape == (37 * 160,) and np.isfinite(y).all() and np.abs(y).max() <= 1


def test_causal_model_uses_one_frame_ahead():
    # changing frame t leaves every sample before frame t - 1 (minus the iSTFT overlap, 256 samples) unchanged
    G = load(MODEL)
    bits = torch.from_numpy(fec.info48(random_frames(80, 3)).astype(np.float32))[None]
    t = 60
    flipped = bits.clone()
    flipped[0, t] = 1 - flipped[0, t]
    with torch.no_grad():
        torch.manual_seed(0)
        y0 = G(bits)[0]
        torch.manual_seed(0)
        y1 = G(flipped)[0]
    first = int(torch.nonzero((y1 - y0).abs() > 1e-6).min())
    assert 160 * (t - 1) - 256 <= first < 160 * t


@pytest.mark.parametrize("model", [MODEL, LARGE])
def test_chunked_equals_whole(model):
    G = load(model)
    info = fec.info48(random_frames(300, 1))
    whole = decode(G, info, chunk_frames=1500)
    chunked = decode(G, info, chunk_frames=100, context=200)  # context covers the receptive field
    # samples differ only by the excitation's noise draw (as with another seed); the frame energies agree
    level = lambda y: 10 * np.log10((y.reshape(-1, 160) ** 2).mean(1) + 1e-9)
    assert np.median(np.abs(level(whole) - level(chunked))) < 1.0


def test_hard_soft_and_erasures():
    cw = random_frames(20, 2)
    noisy = cw.copy()
    noisy[:, fec.GOLAY_A[:3]] ^= 1
    info, nerr = frames.to_info48(noisy)
    assert (info == fec.info48(cw)).all() and (nerr[:, 0] == 3).all()
    erased = np.zeros(20, bool)
    erased[5] = True
    info, _ = frames.to_info48(cw, (cw * 2.0 - 1) * 40, erased)
    assert (info[5] == info[4]).all()


def test_cli_hex(tmp_path):
    src = tmp_path / "frames.hex"
    src.write_text("\n".join([np.packbits(np.array([int(c) for c in TV], np.uint8)).tobytes().hex()] * 12 + ["-"]) + "\n")
    out = tmp_path / "out.wav"
    cli.main([str(src), str(out), "--device", "cpu"])
    y, sr = sf.read(out)
    assert sr == 8000 and len(y) == 13 * 160


def test_captured_samples(tmp_path):
    root = Path(__file__).resolve().parent.parent / "samples"
    for b in sorted(root.glob("*.bin")):
        bits, llr, erased = frames.read_bin(b, b.with_suffix(".llr"))
        assert not erased.any() and (((llr > 0) == (bits > 0)) | (llr == 0)).all()  # LLR signs = received bits
        assert (fec.syndrome(fec.soft_decode(llr)[0]) == 0).all()
    out = tmp_path / "s.wav"
    cli.main([str(root / "vctk_p301_female.bin"), str(out), "--llr", str(root / "vctk_p301_female.llr"), "--device", "cpu"])
    y, sr = sf.read(out)
    assert sr == 8000 and len(y) == 460 * 160 and np.sqrt((y ** 2).mean()) > 0.01
