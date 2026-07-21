// poisson_safety.cpp
// Minimal C++ implementation approximating MATLAB GeneratePoissonSafetyFunction
// Uses finite-difference on a regular grid and Eigen sparse solver.

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>

struct Circle
{
    double cx, cy, r;
};

// Point-in-polygon (ray casting)
bool pointInPoly(const std::vector<std::pair<double, double>> &poly, double x, double y)
{
    bool inside = false;
    int n = (int)poly.size();
    for (int i = 0, j = n - 1; i < n; j = i++)
    {
        double xi = poly[i].first, yi = poly[i].second;
        double xj = poly[j].first, yj = poly[j].second;
        bool intersect = ((yi > y) != (yj > y)) &&
                         (x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi);
        if (intersect)
            inside = !inside;
    }
    return inside;
}

void writeGridCSV(const std::string &name, const Eigen::MatrixXd &M)
{
    std::ofstream f(name);
    for (int i = 0; i < M.rows(); ++i)
    {
        for (int j = 0; j < M.cols(); ++j)
        {
            f << M(i, j);
            if (j + 1 < M.cols())
                f << ',';
        }
        f << '\n';
    }
}

int main()
{
    // Example usage: outer rectangle and one circle hole
    std::vector<std::pair<double, double>> outer = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
    std::vector<Circle> circles = {{0.5, 0.5, 0.2}};

    const int n = 512;
    int nx = n, ny = n;

    double xmin = 0.0, xmax = 1.0;
    double ymin = 0.0, ymax = 1.0;
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(nx, xmin, xmax);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(ny, ymin, ymax);
    double dx = x(1) - x(0);

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> mask(nx, ny);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            double xv = x(i), yv = y(j);
            bool inOuter = pointInPoly(outer, xv, yv);
            bool inAnyCircle = false;
            for (auto &c : circles)
            {
                double dx_ = xv - c.cx, dy_ = yv - c.cy;
                if (std::sqrt(dx_ * dx_ + dy_ * dy_) <= c.r)
                {
                    inAnyCircle = true;
                    break;
                }
            }
            mask(i, j) = inOuter && !inAnyCircle;
        }

    // UX, UY: mimic MATLAB (constant 0.01)
    double ux_val = 0.01, uy_val = 0.01;
    Eigen::MatrixXd ux_grid(nx, ny), uy_grid(nx, ny), fgrid(nx, ny);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            if (mask(i, j))
            {
                ux_grid(i, j) = ux_val;
                uy_grid(i, j) = uy_val;
                fgrid(i, j) = std::sqrt(ux_val * ux_val + uy_val * uy_val);
            }
            else
            {
                ux_grid(i, j) = 0.0;
                uy_grid(i, j) = 0.0;
                fgrid(i, j) = 0.0;
            }
        }

    // Identify unknowns: nodes inside mask that are NOT boundary (all 4 neighbors inside)
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> idx(nx, ny);
    idx.setConstant(-1);
    int count = 0;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
        {
            if (!mask(i, j))
                continue;
            bool allNeigh = mask(i - 1, j) && mask(i + 1, j) && mask(i, j - 1) && mask(i, j + 1);
            if (allNeigh)
            {
                idx(i, j) = count++;
            }
        }

    typedef Eigen::Triplet<double> T;
    std::vector<T> trip;
    Eigen::VectorXd b = Eigen::VectorXd::Zero(count);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
        {
            int id = idx(i, j);
            if (id < 0)
                continue;
            // 4*h_ij - sum_neighbors = f * dx^2
            double rhs = fgrid(i, j) * dx * dx * 1.0;
            trip.emplace_back(id, id, 4.0);
            // neighbors
            const int di[4] = {-1, 1, 0, 0};
            const int dj[4] = {0, 0, -1, 1};
            for (int k = 0; k < 4; ++k)
            {
                int ni = i + di[k], nj = j + dj[k];
                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny)
                    continue;
                if (idx(ni, nj) >= 0)
                {
                    trip.emplace_back(id, idx(ni, nj), -1.0);
                }
                else
                {
                    // neighbor is boundary or outside domain: Dirichlet zero contributes nothing
                }
            }
            b(id) = rhs;
        }

    Eigen::VectorXd xsol = Eigen::VectorXd::Zero(count);
    if (count > 0)
    {
        Eigen::SparseMatrix<double> A(count, count);
        A.setFromTriplets(trip.begin(), trip.end());
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        xsol = solver.solve(b);
    }

    Eigen::MatrixXd h(nx, ny);
    h.setZero();
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            if (idx(i, j) >= 0)
                h(i, j) = xsol(idx(i, j));
            else
                h(i, j) = 0.0; // boundaries and outside -> zero
        }

    // gradients (central differences)
    Eigen::MatrixXd dhdx(nx, ny), dhdy(nx, ny);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
        {
            dhdx(i, j) = (h(i + 1, j) - h(i - 1, j)) / (2 * dx);
            dhdy(i, j) = (h(i, j + 1) - h(i, j - 1)) / (2 * dx);
        }
    // edges: forward/backward
    for (int i = 0; i < nx; ++i)
    {
        dhdx(i, 0) = (h(i, 1) - h(i, 0)) / dx;
        dhdx(i, ny - 1) = (h(i, ny - 1) - h(i, ny - 2)) / dx;
    }
    for (int j = 0; j < ny; ++j)
    {
        dhdy(0, j) = (h(1, j) - h(0, j)) / dx;
        dhdy(nx - 1, j) = (h(nx - 1, j) - h(nx - 2, j)) / dx;
    }

    // Write outputs
    writeGridCSV("h.csv", h.transpose());
    writeGridCSV("dhdx.csv", dhdx.transpose());
    writeGridCSV("dhdy.csv", dhdy.transpose());
    writeGridCSV("ux.csv", ux_grid.transpose());
    writeGridCSV("uy.csv", uy_grid.transpose());

    std::cout << "Wrote h.csv, dhdx.csv, dhdy.csv, ux.csv, uy.csv\n";
    return 0;
}
