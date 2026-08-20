#pragma once

#include <Eigen/Dense>
#include <utility>
#include <vector>

struct Circle
{
    double cx, cy, r;
};

struct SolverConfig
{
    int nx = 512;
    int ny = 512;
    double xmin = 0.0;
    double xmax = 1.0;
    double ymin = 0.0;
    double ymax = 1.0;
    double ux_val = 0.01;
    double uy_val = 0.01;
    std::vector<std::pair<double, double>> outer_boundary = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
    std::vector<Circle> circles;
};

struct PoissonResult
{
    Eigen::MatrixXd h;
    Eigen::MatrixXd dhdx;
    Eigen::MatrixXd dhdy;
    Eigen::MatrixXd ux_grid;
    Eigen::MatrixXd uy_grid;
};

PoissonResult solvePoissonSafety(const SolverConfig &config);
