"""End-to-end check of build/submission: files in, files out, compared with NumPy in FP64.

Run from the lab directory: python3 tests/check_cli.py
The error bound is the same as in the bench tool: (gamma(2K, FP32) + gamma(K, FP64)) * S,
where S = |A| @ |B| and gamma(q, u) = q * u / (1 - q * u).
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SUBMISSION = Path("build/submission")
SEED = 2026
SHAPES = [(1, 1, 1), (33, 40, 31), (127, 65, 257), (300, 200, 100)]
RUNS = [
    ("naive", []),
    ("tiled", []),
    ("tiled", ["--tile", "8"]),
    ("tiled", ["--tile", "32"]),
    ("cublas", []),
]


def gamma(q, unit_roundoff):
    qu = q * unit_roundoff
    return qu / (1.0 - qu)


def write_input(path, a, b):
    with open(path, "w") as f:
        f.write(f"{a.shape[0]} {a.shape[1]} {b.shape[1]}\n")
        np.savetxt(f, a, fmt="%.9g")
        np.savetxt(f, b, fmt="%.9g")


def read_output(path, m, n):
    with open(path) as f:
        header = f.readline().split()
        values = np.loadtxt(f, dtype=np.float64, ndmin=2)
    assert header == [str(m), str(n)], f"bad header {header}"
    assert values.shape == (m, n), f"bad shape {values.shape}"
    return values


def check(workdir, a, b, mode, extra):
    m, k = a.shape
    n = b.shape[1]
    input_path = workdir / "in.txt"
    output_path = workdir / "out.txt"
    write_input(input_path, a, b)

    cmd = [str(SUBMISSION), "--input", str(input_path), "--output", str(output_path)]
    subprocess.run(cmd + ["--mode", mode] + extra, check=True, capture_output=True)

    reference = a.astype(np.float64) @ b.astype(np.float64)
    sums = np.abs(a).astype(np.float64) @ np.abs(b).astype(np.float64)
    bound = (gamma(2 * k, 2.0**-24) + gamma(k, 2.0**-53)) * sums
    error = np.abs(read_output(output_path, m, n) - reference)
    return error.max(), int((error > bound).sum())


def main():
    rng = np.random.default_rng(SEED)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for shape in SHAPES:
            m, k, n = shape
            a = rng.uniform(-1, 1, (m, k)).astype(np.float32)
            b = rng.uniform(-1, 1, (k, n)).astype(np.float32)
            for mode, extra in RUNS:
                max_error, bad = check(Path(tmp), a, b, mode, extra)
                failures += bad > 0
                label = " ".join([mode] + extra)
                print(f"{shape} {label}: max error {max_error:.3e}, over bound {bad}")
    print("failures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
