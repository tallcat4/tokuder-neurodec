"""Readers for TOKUDER TCH voice frames (72 bits / 20 ms, as received, in air order) -> 48 voice bits.

Every reader returns (bits72 [T, 72] uint8, llr [T, 72] float32 or None, erased [T] bool). `to_info48`
corrects them with the recovered channel code (soft ML decoding when LLRs are given, otherwise hard
decoding) and replaces erased frames by the previous frame, as the decoder was trained.
"""
import json
from pathlib import Path

import numpy as np

from . import fec

N_BITS = 72
SLOTS = 4  # TCH voice frames per 80 ms RF frame
RF_FRAME = 0.08


def read_bin(path, llr_path=None):
    """9 bytes per frame (72 bits, MSB first). Optional int8 LLR file [T, 72] (positive = 1, 0 = erased)."""
    raw = np.fromfile(path, np.uint8)
    if len(raw) % 9:
        raise ValueError(f"{path}: {len(raw)} bytes is not a whole number of 9-byte frames")
    bits = np.unpackbits(raw).reshape(-1, N_BITS)
    llr = None
    if llr_path is not None:
        llr = np.fromfile(llr_path, np.int8).astype(np.float32)
        if llr.size != bits.size:
            raise ValueError(f"{llr_path}: {llr.size} values for {len(bits)} frames")
        llr = llr.reshape(-1, N_BITS)
    erased = (np.abs(llr).sum(1) == 0) if llr is not None else np.zeros(len(bits), bool)
    return bits, llr, erased


def read_hex(path):
    """One frame per line as 18 hex digits (spaces allowed); an empty line or '-' is an erased frame."""
    bits, erased = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        h = line.split("#")[0].replace(" ", "").strip()
        if h in ("", "-"):
            bits.append(np.zeros(N_BITS, np.uint8)); erased.append(True)
            continue
        if len(h) != 18:
            raise ValueError(f"{path}: expected 18 hex digits per frame, got {h!r}")
        bits.append(np.unpackbits(np.frombuffer(bytes.fromhex(h), np.uint8))); erased.append(False)
    return np.array(bits, np.uint8).reshape(-1, N_BITS), None, np.array(erased, bool)


def read_t99(log_dir):
    """A receiver log directory (rf.jsonl, rf.llr): TCH halves of each 80 ms RF frame in time
    order, frame 0 = slot 0 of the first voice RF frame; missing RF frames / FACCH halves are erased."""
    a = Path(log_dir)
    rf = [json.loads(l) for l in (a / "rf.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    llr_file = np.fromfile(a / "rf.llr", np.int8).reshape(-1, SLOTS, N_BITS) if (a / "rf.llr").exists() else None
    voice = [f for f in rf if any(h["kind"] == "TCH" for h in f["halves"])]
    if not voice:
        raise ValueError(f"{a}: no TCH frames")
    t0 = voice[0]["t"]
    T = (int(round((voice[-1]["t"] - t0) / RF_FRAME)) + 1) * SLOTS
    bits = np.zeros((T, N_BITS), np.uint8)
    llr = np.zeros((T, N_BITS), np.float32)
    erased = np.ones(T, bool)
    for f in voice:
        k = int(round((f["t"] - t0) / RF_FRAME))
        for half, h in enumerate(f["halves"]):
            if h["kind"] != "TCH":
                continue
            j = k * SLOTS + 2 * half
            bits[j: j + 2] = np.unpackbits(np.frombuffer(bytes.fromhex(h["voice"].replace(" ", "")), np.uint8)).reshape(2, N_BITS)
            erased[j: j + 2] = False
            if llr_file is not None and f.get("burst") is not None:
                llr[j: j + 2] = llr_file[f["burst"], 2 * half: 2 * half + 2]
            else:  # no soft values for this half: hard decision
                llr[j: j + 2] = np.where(bits[j: j + 2] > 0, 64, -64)
    return bits, llr, erased


def to_info48(bits, llr=None, erased=None):
    """-> (info [T, 48] uint8, corrected Golay words per frame [T, 2])."""
    if llr is not None:
        fixed, nerr = fec.soft_decode(llr)
    else:
        fixed, nerr = fec.hard_decode(bits)
    info = fec.info48(fixed).astype(np.uint8)
    if erased is not None:
        for j in range(len(info)):
            if erased[j]:
                info[j] = info[j - 1] if j else 0
    return info, nerr
