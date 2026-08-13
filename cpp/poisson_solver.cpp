#include "poisson_solver.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{

    bool pointInPoly(const std::vector<std::pair<double, double>> &poly, double x, double y)
    {
        bool inside = false;
        int n = static_cast<int>(poly.size());
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

} // namespace

PoissonResult solvePoissonSafety(const SolverConfig &config)
{
    if (config.nx < 3 || config.ny < 3)
    {
        throw std::invalid_argument("nx and ny must be at least 3");
    }

    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(config.nx, config.xmin, config.xmax);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(config.ny, config.ymin, config.ymax);
    double dx = x(1) - x(0);

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> mask(config.nx, config.ny);
    for (int i = 0; i < config.nx; ++i)
        for (int j = 0; j < config.ny; ++j)
        {
            double xv = x(i), yv = y(j);
            bool inOuter = pointInPoly(config.outer_boundary, xv, yv);
            bool inAnyCircle = false;
            for (const auto &c : config.circles)
            {
                double dx_ = xv - c.cx;
                double dy_ = yv - c.cy;
                double dist2 = dx_ * dx_ + dy_ * dy_;
                if (dist2 <= c.r * c.r)
                {
                    inAnyCircle = true;
                    break;
                }
            }
            mask(i, j) = inOuter && !inAnyCircle;
        }

    Eigen::MatrixXd ux_grid(config.nx, config.ny);
    Eigen::MatrixXd uy_grid(config.nx, config.ny);
    Eigen::MatrixXd fgrid(config.nx, config.ny);
    double fval = std::sqrt(config.ux_val * config.ux_val + config.uy_val * config.uy_val);
    for (int i = 0; i < config.nx; ++i)
        for (int j = 0; j < config.ny; ++j)
        {
            if (mask(i, j))
            {
                ux_grid(i, j) = config.ux_val;
                uy_grid(i, j) = config.uy_val;
                fgrid(i, j) = fval;
            }
            else
            {
                ux_grid(i, j) = 0.0;
                uy_grid(i, j) = 0.0;
                fgrid(i, j) = 0.0;
            }
        }

    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> idx(config.nx, config.ny);
    idx.setConstant(-1);
    int count = 0;
    for (int i = 1; i < config.nx - 1; ++i)
        for (int j = 1; j < config.ny - 1; ++j)
        {
            if (!mask(i, j))
                continue;
            bool allNeigh = mask(i - 1, j) && mask(i + 1, j) && mask(i, j - 1) && mask(i, j + 1);
            if (allNeigh)
            {
                idx(i, j) = count++;
            }
        }

    std::vector<Eigen::Triplet<double>> trip;
    Eigen::VectorXd b = Eigen::VectorXd::Zero(count);
    trip.reserve(static_cast<size_t>(count) * 5);
    for (int i = 1; i < config.nx - 1; ++i)
        for (int j = 1; j < config.ny - 1; ++j)
        {
            int id = idx(i, j);
            if (id < 0)
                continue;

            double rhs = fgrid(i, j) * dx * dx;
            trip.emplace_back(id, id, 4.0);

            const int di[4] = {-1, 1, 0, 0};
            const int dj[4] = {0, 0, -1, 1};
            for (int k = 0; k < 4; ++k)
            {
                int ni = i + di[k];
                int nj = j + dj[k];
                if (ni < 0 || ni >= config.nx || nj < 0 || nj >= config.ny)
                    continue;
                if (idx(ni, nj) >= 0)
                {
                    trip.emplace_back(id, idx(ni, nj), -1.0);
                }
            }
            b(id) = rhs;
        }

    Eigen::VectorXd xsol = Eigen::VectorXd::Zero(count);
    if (count > 0)
    {
        Eigen::SparseMatrix<double> A(count, count);
        A.setFromTriplets(trip.begin(), trip.end());
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("SimplicialLLT decomposition failed");
        }
        xsol = solver.solve(b);
    }

    Eigen::MatrixXd h(config.nx, config.ny);
    h.setZero();
    for (int i = 0; i < config.nx; ++i)
        for (int j = 0; j < config.ny; ++j)
        {
            if (idx(i, j) >= 0)
                h(i, j) = xsol(idx(i, j));
            else
                h(i, j) = 0.0;
        }

    Eigen::MatrixXd dhdx(config.nx, config.ny);
    Eigen::MatrixXd dhdy(config.nx, config.ny);
    for (int i = 1; i < config.nx - 1; ++i)
        for (int j = 1; j < config.ny - 1; ++j)
        {
            dhdx(i, j) = (h(i + 1, j) - h(i - 1, j)) / (2 * dx);
            dhdy(i, j) = (h(i, j + 1) - h(i, j - 1)) / (2 * dx);
        }
    for (int i = 0; i < config.nx; ++i)
    {
        dhdx(i, 0) = (h(i, 1) - h(i, 0)) / dx;
        dhdx(i, config.ny - 1) = (h(i, config.ny - 1) - h(i, config.ny - 2)) / dx;
    }
    for (int j = 0; j < config.ny; ++j)
    {
        dhdy(0, j) = (h(1, j) - h(0, j)) / dx;
        dhdy(config.nx - 1, j) = (h(config.nx - 1, j) - h(config.nx - 2, j)) / dx;
    }

    return {h, dhdx, dhdy, ux_grid, uy_grid};
}
