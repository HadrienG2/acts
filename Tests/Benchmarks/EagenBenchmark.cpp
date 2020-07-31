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

    // TODO: Should test row vectors and "middle segment" scenario as well
    if constexpr ((ROWS > 1) && (COLS == 1)) {
        bench("extractHead",
              [&] { return eagenMat.template extractHead<ROWS-1>(); },
              [&] { return eigenMat.template head<ROWS-1>().eval(); },
              ITERS_LINEAR);
        bench("extractTail",
              [&] { return eagenMat.template extractTail<ROWS-1>(); },
              [&] { return eigenMat.template tail<ROWS-1>().eval(); },
              ITERS_LINEAR);

        using EagenSegment = Eagen::Vector<double, ROWS-1>;
        bench("zeroHead",
              [&] { eagenMat.template setHead<ROWS-1>(EagenSegment::Zero()); },
              [&] { eigenMat.template head<ROWS-1>().setZero(); },
              ITERS_LINEAR);
        bench("zeroTail",
              [&] { eagenMat.template setTail<ROWS-1>(EagenSegment::Zero()); },
              [&] { eigenMat.template tail<ROWS-1>().setZero(); },
              ITERS_LINEAR);

        bench("transferHead",
              [&] {
                  eagenMat.template setHead<ROWS-1>(
                      eagenMat2.template extractHead<ROWS-1>()
                  );
              },
              [&] {
                  eigenMat.template head<ROWS-1>() =
                      eigenMat2.template head<ROWS-1>();
              },
              ITERS_LINEAR);
        bench("transferTail",
              [&] {
                  eagenMat.template setTail<ROWS-1>(
                      eagenMat2.template extractTail<ROWS-1>()
                  );
              },
              [&] {
                  eigenMat.template tail<ROWS-1>() =
                      eigenMat2.template tail<ROWS-1>();
              },
              ITERS_LINEAR);
    }

    if constexpr ((ROWS > 1) && (COLS > 1)) {
        bench("extractTopLeftCorner",
              [&] { return eagenMat.template extractTopLeftCorner<ROWS-1,
                                                                  COLS-1>(); },
              [&] { return eigenMat.template topLeftCorner<ROWS-1,
                                                           COLS-1>().eval(); },
              ITERS_LINEAR);

        using EagenCorner = Eagen::Matrix<double, ROWS-1, COLS-1>;
        bench("zeroTopLeftCorner",
              [&] {
                  eagenMat.template setTopLeftCorner<ROWS-1, COLS-1>(
                      EagenCorner::Zero()
                  );
              },
              [&] {
                  eigenMat.template topLeftCorner<ROWS-1, COLS-1>()
                          .setZero();
              },
              ITERS_LINEAR);

        bench("transferTopLeftCorner",
              [&] {
                  eagenMat.template setTopLeftCorner<ROWS-1, COLS-1>(
                      eagenMat2.template extractTopLeftCorner<ROWS-1, COLS-1>()
                  );
              },
              [&] {
                  eigenMat.template topLeftCorner<ROWS-1, COLS-1>() =
                      eigenMat2.template topLeftCorner<ROWS-1, COLS-1>();
              },
              ITERS_LINEAR);
    }

    if constexpr (ROWS > 1) {
        bench("extractFirstRow",
              [&] { return eagenMat.extractRow(0); },
              [&] { return eigenMat.row(0).eval(); },
              ITERS_LINEAR);
        bench("extractLastRow",
              [&] { return eagenMat.extractRow(ROWS-1); },
              [&] { return eigenMat.row(ROWS-1).eval(); },
              ITERS_LINEAR);

        using EagenRow = Eagen::RowVector<double, COLS>;
        bench("zeroFirstRow",
              [&] { eagenMat.setRow(0, EagenRow::Zero()); },
              [&] { eigenMat.row(0).setZero(); },
              ITERS_LINEAR);
        bench("zeroLastRow",
              [&] { eagenMat.setRow(ROWS-1, EagenRow::Zero()); },
              [&] { eigenMat.row(ROWS-1).setZero(); },
              ITERS_LINEAR);

        bench("transferFirstRow",
              [&] { eagenMat.setRow(0, eagenMat2.extractRow(0)); },
              [&] { eigenMat.row(0) = eigenMat2.row(0); },
              ITERS_LINEAR);
        bench("transferLastRow",
              [&] { eagenMat.setRow(ROWS-1, eagenMat2.extractRow(ROWS-1)); },
              [&] { eigenMat.row(ROWS-1) = eigenMat2.row(ROWS-1); },
              ITERS_LINEAR);
    }

    if constexpr (COLS > 1) {
        bench("extractFirstCol",
              [&] { return eagenMat.extractCol(0); },
              [&] { return eigenMat.col(0).eval(); },
              ITERS_LINEAR);
        bench("extractLastCol",
              [&] { return eagenMat.extractCol(COLS-1); },
              [&] { return eigenMat.col(COLS-1).eval(); },
              ITERS_LINEAR);

        using EagenCol = Eagen::Vector<double, ROWS>;
        bench("zeroFirstCol",
              [&] { eagenMat.setCol(0, EagenCol::Zero()); },
              [&] { eigenMat.col(0).setZero(); },
              ITERS_LINEAR);
        bench("zeroLastCol",
              [&] { eagenMat.setCol(COLS-1, EagenCol::Zero()); },
              [&] { eigenMat.col(COLS-1).setZero(); },
              ITERS_LINEAR);

        bench("transferFirstCol",
              [&] { eagenMat.setCol(0, eagenMat2.extractCol(0)); },
              [&] { eigenMat.col(0) = eigenMat2.col(0); },
              ITERS_LINEAR);
        bench("transferLastCol",
              [&] { eagenMat.setCol(COLS-1, eagenMat2.extractCol(COLS-1)); },
              [&] { eigenMat.col(COLS-1) = eigenMat2.col(COLS-1); },
              ITERS_LINEAR);
    }

    // TODO: Should test "middle blocks" as well
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
    /* matrixBenchmarkLoop<8>(); */
    matrixBenchmark<8, 1>();
    matrixBenchmark<8, 8>();
}