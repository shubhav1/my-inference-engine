# Run the engine on a Modal GPU. Weights fetched once and stored in Modal Volume.
# uv run modal run modal_run.py::download_weights   # one-time: pull Qwen2.5-1.5B into the volume
# uv run modal run modal_run.py::test_fa2           # FA2 kernel vs SDPA correctness check
# uv run modal run modal_run.py::benchmark --description "..."  # all 4 benchmarks, rows land in metrics/*.md

import modal

app = modal.App("my-inference-engine")
weights = modal.Volume.from_name("qwen-weights", create_if_missing=True)
# Triton's compiled kernels + autotune winners, kept across runs so the config search only happens once
triton_cache = modal.Volume.from_name("triton-cache", create_if_missing=True)

WEIGHTS_DIR = "/weights/qwen2.5-1.5b"


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


@app.function(image=gpu_image, gpu="A10G", volumes={"/triton-cache": triton_cache}, timeout=3600)
def test_fa2():
    import os
    import subprocess

    env = {**os.environ, "TRITON_CACHE_DIR": "/triton-cache", "TRITON_PRINT_AUTOTUNING": "1"}
    try:
        subprocess.run(["python", "kernels/test_fa2.py"], cwd="/root/app", env=env, check=True)
    finally:
        triton_cache.commit()


BENCHMARKS = ["benchmark.py", "benchmark_hf.py", "benchmark_concurrency.py", "benchmark_concurrency_hf.py"]
RESULTS_FILES = ["results.md", "concurrency_results.md"]


@app.function(image=gpu_image, gpu="A10G", volumes={"/weights": weights, "/triton-cache": triton_cache}, timeout=3600)
def run_benchmarks(description: str) -> dict:
    """Run every benchmark (ours and HF, single chat and concurrent) on one GPU, back to back, and
    return the rows each results file gained."""
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
        for script in BENCHMARKS:
            print(f"\n===== {script} =====", flush=True)
            subprocess.run(["python", f"metrics/{script}", description], cwd="/tmp/app", env=env, check=True)
    finally:
        triton_cache.commit()
    return {f: open(f"{metrics}/{f}").readlines()[before[f]:] for f in RESULTS_FILES}


@app.local_entrypoint()
def benchmark(description: str):
    #   uv run modal run modal_run.py::benchmark --description "what changed"
    from pathlib import Path

    new_rows = run_benchmarks.remote(description)
    for name, rows in new_rows.items():
        path = Path(__file__).parent / "metrics" / name
        with open(path, "a") as f:
            f.writelines(rows)
        print(f"\n{len(rows)} new row(s) appended to {path}:")
        print("".join(rows), end="")
