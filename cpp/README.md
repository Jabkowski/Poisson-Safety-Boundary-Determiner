# Poisson Safety (C++ implementation)

This folder contains a minimal C++ implementation approximating the MATLAB `GeneratePoissonSafetyFunction` using a finite-difference solver on a regular grid and Eigen's sparse solver.

Requirements
- Eigen (header-only). On Debian/Ubuntu: `sudo apt install libeigen3-dev`

Compile

```bash
g++ -O2 -std=c++17 main.cpp poisson_solver.cpp -I/usr/include/eigen3 -o poisson_safety
```

Run

```bash
./poisson_safety
```

This writes `h.csv`, `dhdx.csv`, `dhdy.csv`, `ux.csv`, `uy.csv` into the current directory.

Tests
----

To compile and run the example, then execute the Python test:

```bash
bash test/run_test.sh
python3 test/test_poisson.py
```

Notes
- This implementation uses a rectangular bounding box and ray-casting for polygon inclusion.
- It mimics the MATLAB script's effective behavior (constant boundary `u = [0.01,0.01]`) and solves
  the Poisson equation on the masked domain with Dirichlet zero on boundaries.
