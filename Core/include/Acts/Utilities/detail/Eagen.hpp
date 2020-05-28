// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <utility>

// for GNU: ignore this specific warning, otherwise just include Eigen/Dense
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include <Eigen/Dense>
#pragma GCC diagnostic pop
#else
#include <Eigen/Dense>
#endif

namespace Acts {

namespace detail {

/// Eagerly evaluated variant of Eigen, without expression templates
namespace Eagen {

template <typename Scalar, int Rows, int Cols,
          int Options = Eigen::AutoAlign |
                        ( (Rows==1 && Cols!=1) ? Eigen::RowMajor
                          : (Cols==1 && Rows!=1) ? Eigen::ColMajor
                          : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
          int MaxRows = Rows,
          int MaxCols = Cols>
class Matrix {
public:
    // === Eigen::Matrix interface ===

    // Basic lifecycle
    Matrix() = default;
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;
#if EIGEN_HAS_RVALUE_REFERENCES
    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;
#endif

    // Build and assign from anything Eigen supports building or assigning from
    template <typename... ArgTypes>
    Matrix(ArgTypes&&... args) : m_inner(std::forward(args)...) {}
    template <typename Other>
    Matrix& operator=(Other&& other) {
        m_inner = std::forward(other);
        return *this;
    }

    // Emulate Eigen::Matrix's base class typedef
    using Base = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;

    // TODO: Replicate interface of Eigen::PlainObjectBase
    // TODO: Replicate interface of Eigen::DenseBase
    // TODO: Replicate interface of all Eigen::DenseCoeffsBase types
    // TODO: Replicate interface of Eigen::EigenBase

private:
    Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxRows> m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
