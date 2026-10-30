"""TOKUDER (ARIB STD-T99 / DRCR) neural decoder: TCH voice frames -> speech."""
from .model import Decoder, decode, load

__all__ = ["Decoder", "decode", "load"]
