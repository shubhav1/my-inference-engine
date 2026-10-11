# Run the engine on a Modal GPU. Weights fetched once and stored in Modal Volume.
# uv run modal run modal_run.py::download_weights   # one-time: pull Qwen2.5-1.5B into the volume
# uv run modal run modal_run.py::test_fa2           # FA2 kernel vs SDPA correctness check
# uv run modal run modal_run.py::benchmark --description "..."  # our engine, rows appended to metrics/*.md
# uv run modal run modal_run.py::baseline          # HF reference, rows go at the top of metrics/*.md

import modal

app = modal.App("my-inference-engine")
weights = modal.Volume.from_name("qwen-weights", create_if_missing=True)
# Triton's compiled kernels + autotune winners, kept across runs so the config search only happens once
triton_cache = modal.Volume.from_name("triton-cache", create_if_missing=True)

WEIGHTS_DIR = "/weights/qwen2.5-1.5b"

# pinned so every run lands on the same hardware: unpinned, "A10G" can be served by other clouds'
# NVIDIA A10 (different card + host CPU), which moved results by ~25% between runs
GPU = "A10G"
GPU_CLOUD = "aws"


@app.function(
    image=modal.Image.debian_slim(python_version="3.13").pip_install("huggingface-hub"),
    volumes={"/weights": weights},
    timeout=900,
)
def download_weights():
    import hashlib

    from huggingface_hub import snapshot_download

    snapshot_download("Qwen/Qwen2.5-1.5B", local_dir=WEIGHTS_DIR)
    weights.commit()

    # compare against the local copy: shasum -a 256 qwen2.5-1.5b/model.safetensors
    h = hashlib.sha256()
    with open(f"{WEIGHTS_DIR}/model.safetensors", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    print("model.safetensors sha256:", h.hexdigest())


# project code + its runtime deps (torch's linux wheel brings CUDA and triton)
gpu_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch>=2.14.0", "safetensors>=0.8.0", "regex>=2026.9.10", "fastapi>=0.141.1",
                 "python-multipart>=0.0.32", "transformers>=5.17.0", "uvicorn>=0.53.0")
    .add_local_dir(".", remote_path="/root/app", ignore=[".venv", ".git", "qwen2.5-1.5b", "**/__pycache__"])
)


@app.function(image=gpu_image, gpu=GPU, cloud=GPU_CLOUD, volumes={"/triton-cache": triton_cache}, timeout=3600)
def test_fa2():
    import os
    import subprocess

    env = {**os.environ, "TRITON_CACHE_DIR": "/triton-cache", "TRITON_PRINT_AUTOTUNING": "1"}
    try:
        subprocess.run(["python", "kernels/test_fa2.py"], cwd="/root/app", env=env, check=True)
    finally:
        triton_cache.commit()


OURS = ["benchmark.py", "benchmark_concurrency.py"]
HF_BASELINE = ["benchmark_hf.py", "benchmark_concurrency_hf.py"]
RESULTS_FILES = ["results.md", "concurrency_results.md"]


@app.function(image=gpu_image, gpu=GPU, cloud=GPU_CLOUD, volumes={"/weights": weights, "/triton-cache": triton_cache}, timeout=3600)
def run_benchmarks(scripts: list, description: str) -> dict:
    """Run the given benchmark scripts on one GPU, back to back, and return the rows each results
    file gained."""
    import os
    import shutil
    import subprocess

    # private writable copy of the code, with the weights where chat.py expects them
    shutil.copytree("/root/app", "/tmp/app")
    os.symlink(WEIGHTS_DIR, "/tmp/app/qwen2.5-1.5b")
    metrics = "/tmp/app/metrics"
    before = {f: len(open(f"{metrics}/{f}").readlines()) for f in RESULTS_FILES}

    env = {**os.environ, "TRITON_CACHE_DIR": "/triton-cache"}
    try:
        for script in scripts:
            print(f"\n===== {script} =====", flush=True)
            subprocess.run(["python", f"metrics/{script}", description], cwd="/tmp/app", env=env, check=True)
    finally:
        triton_cache.commit()
    return {f: open(f"{metrics}/{f}").readlines()[before[f]:] for f in RESULTS_FILES}


def _save_rows(new_rows, at_top):
    """Add each file's new rows locally: appended, or right under the table header (baseline)."""
    from pathlib import Path

    for name, rows in new_rows.items():
        path = Path(__file__).parent / "metrics" / name
        lines = path.read_text().splitlines(keepends=True)
        if at_top:
            header_end = next(i for i, line in enumerate(lines) if line.startswith("|---")) + 1
            lines[header_end:header_end] = rows
        else:
            lines += rows
        path.write_text("".join(lines))
        print(f"\n{len(rows)} new row(s) {'at the top of' if at_top else 'appended to'} {path}:")
        print("".join(rows), end="")


@app.local_entrypoint()
def benchmark(description: str):
    #   uv run modal run modal_run.py::benchmark --description "what changed"
    _save_rows(run_benchmarks.remote(OURS, description), at_top=False)


@app.local_entrypoint()
def baseline():
    #   uv run modal run modal_run.py::baseline   # HF transformers reference, run once per GPU
    _save_rows(run_benchmarks.remote(HF_BASELINE, "A10G baseline, same dtype + KV budget"), at_top=True)
