"""The neural TOKUDER decoder: 48 voice bits per 20 ms frame -> 8 kHz speech.

bit-token frontend (per-frame MLP + ConvNeXt blocks over frames; predicts 80-band log-mel, log F0,
voicing and band aperiodicity) -> mel bridge -> small iSTFT (Vocos-style) generator with
harmonic-plus-noise excitation. Inference-only copy of the architecture trained with
https://github.com/tallcat4/neurodec (parameter names are kept: the "G" state dict of its checkpoints loads unchanged).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .frontends import BitTokenFrontend
from .istft import ISTFTGenerator, conv1d

FRAME = 160  # samples per 20 ms frame at 8 kHz
SR = 8000


class Decoder(nn.Module):
    def __init__(self, model_cfg: dict, n_bits: int = 48):
        super().__init__()
        m = model_cfg
        causal = bool(m.get("causal", False))
        self.block_frames = int(m.get("block_frames", 1))
        self.frontend = BitTokenFrontend(n_bits, block_frames=self.block_frames, out_dim=m["dequant"]["ffn_dims"][-1],
                                         aux_mels=int(m["aux_mels"]), f0_head=bool(m.get("f0_head", False)),
                                         ap_bands=int(m.get("ap_bands", 0)), causal=causal,
                                         lookahead=int(m.get("lookahead_frames", 0)), **m.get("bittoken", {}))
        self.bridge = conv1d(int(m["aux_mels"]), self.frontend.out_dim, 5, causal)
        self.generator = ISTFTGenerator(self.frontend.out_dim, samples_per_frame=FRAME, causal=causal, **m["istft"])

    @torch.no_grad()
    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """bits [B, T, 48] (0/1, frame 0 = first TCH slot of an 80 ms RF frame) -> wav [B, 160 T]."""
        _, mel, h = self.frontend.forward_with_aux(bits, with_f0=True)
        h = h.float()
        f0, voiced = h[:, 0].exp(), (torch.sigmoid(h[:, 1]) > 0.5).float()
        ap = torch.sigmoid(h[:, 2:]).transpose(1, 2) if h.shape[1] > 2 else None
        return self.generator(self.bridge(mel), f0=f0, voiced=voiced, aperiodicity=ap)


def load(model_path: str | Path, device: str = "cpu") -> Decoder:
    """models/<name>.pt (state dict) + models/<name>.json (architecture)."""
    model_path = Path(model_path)
    cfg = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    G = Decoder(cfg["model"], cfg.get("n_bits", 48))
    G.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return G.to(device).eval()


def decode(G: Decoder, bits: np.ndarray, device: str = "cpu", chunk_frames: int = 1500, context: int = 52,
           seed: int = 0) -> np.ndarray:
    """bits [T, 48] -> float32 wav [160 T] in [-1, 1]. Long inputs are decoded in chunks with `context`
    frames of overlap on both sides; chunk boundaries stay on RF-frame boundaries (multiples of 4)."""
    g = G.block_frames
    T = bits.shape[0]
    b = np.pad(bits, ((0, (-T) % g), (0, 0))).astype(np.float32)
    Tp = b.shape[0]
    chunk_frames -= chunk_frames % g
    context -= context % g
    out = np.zeros(Tp * FRAME, np.float32)
    torch.manual_seed(seed)  # the excitation's noise
    for s in range(0, Tp, chunk_frames):
        a, e = max(0, s - context), min(Tp, s + chunk_frames + context)
        y = G(torch.from_numpy(b[a:e])[None].to(device))[0].float().cpu().numpy()
        end = min(Tp, s + chunk_frames)
        out[s * FRAME: end * FRAME] = y[(s - a) * FRAME: (end - a) * FRAME]
    return np.clip(out[: T * FRAME], -1, 1)
