// This file is part of the Acts project.
//
// Copyright (C) 2017-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Tests/CommonHelpers/BenchmarkTools.hpp"
#include "Acts/Utilities/detail/Eagen/Eagen.hpp"

#include <iostream>

#include <Eigen/Dense>

using namespace Acts::Test;
namespace Eagen = Acts::detail::Eagen;
using Index = Eigen::Index;

// Benchmark matrices of size ROWSxCOLS
template <Index ROWS, Index COLS>
void matrixBenchmark() {
    std::cout << "=== " << ROWS << 'x' << COLS << " matrices ===" << std::endl
              << std::endl;

    // NOTE: Only benchmarking doubles for now, because benchmarking floats as
    //       well approximately doubles the benchmark's running time and
    //       complexifies usage of constants in benchmark, while usage of floats
    //       remains exceptional in Acts at the moment.
    auto eagenMat = Eagen::Matrix<double, ROWS, COLS>::Zero();
    auto eagenMat2 = eagenMat;
    auto eigenMat = Eigen::Matrix<double, ROWS, COLS>::Zero().eval();
    auto eigenMat2 = eigenMat;

    auto bench = [](const char* title,
                    auto&& eagenMethod,
                    auto&& eigenMethod,
                    int num_iters) {
        std::cout << title << ':' << std::endl;
        std::cout << "- Eagen: " << std::flush
                  << microBenchmark(eagenMethod, num_iters) << std::endl;
        std::cout << "- Eigen: " << std::flush
                  << microBenchmark(eigenMethod, num_iters) << std::endl;
        std::cout << std::endl;
    };

    auto bench_unary = [&](const char* title, auto&& method, int iters) {
        bench(title,
              [&] { return method(eagenMat); },
              [&] { return method(eigenMat); },
              iters);
    };
    auto bench_unary_lazy = [&](const char* title, auto&& method, int iters) {
        bench(title,
              [&] { return method(eagenMat); },
              [&] { return method(eigenMat).eval(); },
              iters);
    };
    auto bench_binary = [&](const char* title, auto&& method, int iters) {
        bench(title,
              [&] { return method(eagenMat, eagenMat2); },
              [&] { return method(eigenMat, eigenMat2); },
              iters);
    };
    auto bench_binary_lazy = [&](const char* title, auto&& method, int iters) {
        bench(title,
              [&] { return method(eagenMat, eagenMat2); },
              [&] { return method(eigenMat, eigenMat2).eval(); },
              iters);
    };

    constexpr int ITERS_PER_ELEM = 60000;
    constexpr int ITERS_LINEAR = ITERS_PER_ELEM / (ROWS * COLS);
    constexpr int ITERS_QUADRATIC = ITERS_LINEAR / (ROWS * COLS);
    constexpr int ITERS_PER_FILL = 10000;

    bench_unary("hasNaN",   
                [](const auto& m) { return m.hasNaN(); },
                ITERS_LINEAR);

    const double cst = 4.2;
    bench_unary("fill",
                [&cst](auto& m) { m.fill(cst); },
                ITERS_PER_FILL);
    bench_unary("setConstant",
                [&cst](auto& m) { m.setConstant(cst); },
                ITERS_PER_FILL);
    bench_unary("setOnes",
                [](auto& m) { m.setOnes(); },
                ITERS_PER_FILL);
    bench_unary("setZero",
                [](auto& m) { m.setZero(); },
                ITERS_PER_FILL);

    bench_unary_lazy("transpose",
                     [](const auto& m) { return m.transpose(); },
                     ITERS_LINEAR);

    bench_unary_lazy("Constant",
                     [&cst](auto m) { return decltype(m)::Constant(cst); },
                     ITERS_LINEAR);
    bench_unary_lazy("Ones",
                     [](auto m) { return decltype(m)::Ones(); },
                     ITERS_LINEAR);
    bench_unary_lazy("Zero",
                     [](auto m) { return decltype(m)::Zero(); },
                     ITERS_LINEAR);

    bench_unary_lazy("operator*(scalar)",
                     [&cst](const auto& m) { return m * cst; },
                     ITERS_LINEAR);
    bench_unary_lazy("operator/(scalar)",
                     [&cst](const auto& m) { return m / cst; },
                     ITERS_LINEAR);

    bench_binary_lazy("operator+(Matrix)",
                      [](const auto& m1, const auto& m2) { return m1 + m2; },
                      ITERS_LINEAR);
    bench_binary_lazy("operator-(Matrix)",
                      [](const auto& m1, const auto& m2) { return m1 - m2; },
                      ITERS_LINEAR);

    bench_binary("operator+=(Matrix)",
                 [](auto& m1, const auto& m2) { m1 += m2; },
                 ITERS_LINEAR);
    bench_binary("operator-=(Matrix)",
                 [](auto& m1, const auto& m2) { m1 -= m2; },
                 ITERS_LINEAR);

    bench_unary_lazy("True M.Mt",
                     [](const auto& m) { return m * m.transpose().eval(); },
                     ITERS_QUADRATIC);
    bench_unary_lazy("Lazy M.Mt",
                     [](const auto& m) { return m * m.transpose(); },
                     ITERS_QUADRATIC);
    bench_unary_lazy("True Mt.M",
                     [](const auto& m) { return m.transpose().eval() * m; },
                     ITERS_QUADRATIC);
    bench_unary_lazy("Lazy Mt.M",
                     [](const auto& m) { return m.transpose() * m; },
                     ITERS_QUADRATIC);
}

// Benchmark matrices of all size from 1x1 to MAX_DIM x MAX_DIM
template <Index MAX_DIM, Index ROWS=1, Index COLS=1>
void matrixBenchmarkLoop() {
    matrixBenchmark<ROWS, COLS>();
    if constexpr (ROWS < MAX_DIM) {
        matrixBenchmarkLoop<MAX_DIM, ROWS+1, COLS>();
    } else if constexpr (COLS < MAX_DIM) {
        matrixBenchmarkLoop<MAX_DIM, 1, COLS+1>();
    }
}

int main(int /*argc*/, char** /*argv[]*/) {
    std::cout << "### Eagen vs Eigen comparative benchmarks ###" << std::endl
              << std::endl;
    // FIXME: Cutting the loop for now to save on build and running time
    /* matrixBenchmarkLoop<4>(); */
    matrixBenchmark<8, 8>();
}