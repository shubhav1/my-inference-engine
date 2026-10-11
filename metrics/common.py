"""Helpers shared by every benchmark script. Importing this also puts the project root (chat.py,
model.py, tokenizer.py) on sys.path, so scripts import it before anything from the engine."""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chat import DEVICE, DTYPE

DTYPE_NAMES = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}


def get_description():
    """The run's description, from the command line (`python metrics/benchmark.py "what changed"`)
    or asked for interactively, made safe to put in a markdown table cell."""
    parser = argparse.ArgumentParser()
    parser.add_argument("description", nargs="?", help="what changed since the last run")
    description = parser.parse_args().description or input("Describe this run: ")
    return description.strip().replace("|", "/").replace("\n", " ")


def device_label():
    """e.g. 'mps fp32' or 'cuda NVIDIA A10G bf16', so rows from different hardware stay comparable."""
    gpu = f" {torch.cuda.get_device_name(DEVICE)}" if DEVICE.type == "cuda" else ""
    return f"{DEVICE.type}{gpu} {DTYPE_NAMES.get(DTYPE, str(DTYPE))}"


def timestamp():
    """'date | time' cells for a results row."""
    return datetime.now().strftime("%Y-%m-%d | %H:%M:%S")
