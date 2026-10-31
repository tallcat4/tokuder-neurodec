"""tokuder2wav: TOKUDER TCH voice frames -> 8 kHz mono 16-bit WAV."""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import frames
from .model import FRAME, SR, decode, load

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "tokuder48_causal_v1.pt"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tokuder2wav", description=__doc__)
    ap.add_argument("input", help="frame file (bin / hex) or a receiver log directory with rf.jsonl / rf.llr (t99)")
    ap.add_argument("output", help="output WAV (8 kHz, mono, 16 bit)")
    ap.add_argument("--format", choices=["bin", "hex", "t99"], default=None,
                    help="bin: 9 bytes per frame; hex: 18 hex digits per line; t99: receiver log directory "
                         "(default: t99 for a directory, hex for .txt/.hex, otherwise bin)")
    ap.add_argument("--llr", default=None, help="bin only: int8 LLRs [T, 72] (positive = 1) for soft decoding")
    ap.add_argument("--slot", type=int, default=0, choices=range(frames.SLOTS),
                    help="TCH slot (0-3) of the first frame inside its 80 ms RF frame (default 0)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    fmt = a.format or ("t99" if Path(a.input).is_dir() else "hex" if Path(a.input).suffix in (".txt", ".hex") else "bin")
    if fmt == "t99":
        bits, llr, erased = frames.read_t99(a.input)
    elif fmt == "hex":
        bits, llr, erased = frames.read_hex(a.input)
    else:
        bits, llr, erased = frames.read_bin(a.input, a.llr)
    if a.llr and fmt != "bin":
        ap.error("--llr applies to --format bin only")
    info, nerr = frames.to_info48(bits, llr, erased)
    # the decoder knows each frame's slot in the RF frame: pad the start so that frame 0 sits at slot 0
    info = np.concatenate([np.repeat(info[:1], a.slot, 0), info]) if a.slot else info
    G = load(a.model, a.device)
    wav = decode(G, info, a.device, seed=a.seed)[a.slot * FRAME:]
    sf.write(a.output, wav, SR, subtype="PCM_16")
    print(f"{len(bits)} frames ({len(bits) * 0.02:.2f} s), erased {int(erased.sum())}, "
          f"corrected Golay words {int((nerr > 0).sum())} -> {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
