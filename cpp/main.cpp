#include "poisson_solver.h"

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>

namespace
{

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

} // namespace

int main()
{
    SolverConfig config;
    config.circles.push_back({0.5, 0.5, 0.1});
    config.circles.push_back({0.25, 0.5, 0.05});
    config.circles.push_back({0.75, 0.5, 0.05});

    const int repetitions = 100;
    std::chrono::milliseconds total = std::chrono::milliseconds(0);

    PoissonResult result{};

    for (int i = 0; i < repetitions; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        result = solvePoissonSafety(config);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        total += duration;
        std::cout << "Iteration " << i << " took " << duration.count() << " ms\n";
    }

    double average = static_cast<double>(total.count()) / repetitions;
    std::cout << "Average duration: " << average << " ms\n";

    writeGridCSV("h.csv", result.h.transpose());
    writeGridCSV("dhdx.csv", result.dhdx.transpose());
    writeGridCSV("dhdy.csv", result.dhdy.transpose());
    writeGridCSV("ux.csv", result.ux_grid.transpose());
    writeGridCSV("uy.csv", result.uy_grid.transpose());

    std::cout << "Wrote h.csv, dhdx.csv, dhdy.csv, ux.csv, uy.csv\n";
    return 0;
}
