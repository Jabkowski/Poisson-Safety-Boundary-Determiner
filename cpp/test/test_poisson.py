#!/usr/bin/env python3
import subprocess
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "poisson_safety"


def run_binary():
    # compile if binary missing
    if not BIN.exists():
        subprocess.check_call(
            ["bash", str(ROOT / "test" / "run_test.sh")],
            cwd=str(ROOT),
        )
    else:
        subprocess.check_call([str(BIN)], cwd=str(ROOT))


def load_csv(name):
    p = ROOT / name
    assert p.exists(), f"Missing {p}"
    return np.loadtxt(p, delimiter=",")


def main():
    run_binary()
    h = load_csv("h.csv")
    dhdx = load_csv("dhdx.csv")
    dhdy = load_csv("dhdy.csv")
    ux = load_csv("ux.csv")
    uy = load_csv("uy.csv")

    # basic checks
    assert h.shape == (128, 128), f"h shape {h.shape}"
    assert dhdx.shape == (128, 128)
    assert dhdy.shape == (128, 128)
    assert ux.shape == (128, 128)
    assert uy.shape == (128, 128)

    assert not np.isnan(h).any(), "NaN in h"
    assert not np.isnan(dhdx).any()

    # ux should have the boundary value 0.01 somewhere
    assert np.isclose(ux.max(), 0.01, atol=1e-6), f"ux.max={ux.max()}"

    print("All tests passed")


if __name__ == "__main__":
    main()
