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

// Propagate some Eigen types and constants
using Index = Eigen::Index;
using NoChange_t = Eigen::NoChange_t;

// Eigen::Matrix, but with eagerly evaluated operations
template <typename Scalar, int Rows, int Cols,
          int Options = Eigen::AutoAlign |
                        ( (Rows==1 && Cols!=1) ? Eigen::RowMajor
                          : (Cols==1 && Rows!=1) ? Eigen::ColMajor
                          : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
          int MaxRows = Rows,
          int MaxCols = Cols>
class Matrix {
public:
    // TODO: If this works, reduce reliance on variadic templates by using the
    //       true method signatures instead.

    // === Eigen::Matrix interface ===

    // Basic lifecycle
    Matrix() = default;
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix(const Matrix<OtherScalar,
                        OtherRows,
                        OtherCols,
                        OtherOptions,
                        OtherMaxRows,
                        OtherMaxCols>& other)
        : m_inner(other.m_inner)
    {}
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix& operator=(const Matrix<OtherScalar,
                                   OtherRows,
                                   OtherCols,
                                   OtherOptions,
                                   OtherMaxRows,
                                   OtherMaxCols>& other) {
        m_inner = other.m_inner;
        return *this;
    }
#if EIGEN_HAS_RVALUE_REFERENCES
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix(Matrix<OtherScalar,
                  OtherRows,
                  OtherCols,
                  OtherOptions,
                  OtherMaxRows,
                  OtherMaxCols>&& other)
        : m_inner(std::move(other.m_inner))
    {}
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix& operator=(Matrix<OtherScalar,
                             OtherRows,
                             OtherCols,
                             OtherOptions,
                             OtherMaxRows,
                             OtherMaxCols>&& other) {
        m_inner = std::move(other.m_inner);
        return *this;
    }
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

    // === Eigen::PlainObjectBase interface ===

    // Coefficient access
    template <typename... Index>
    const Scalar& coeff(Index... indices) const {
        return m_inner.coeff(indices...);
    }
    template <typename... Index>
    const Scalar& coeffRef(Index... indices) const {
        return m_inner.coeffRef(indices...);
    }
    template <typename... Index>
    Scalar& coeffRef(Index... indices) {
        return m_inner.coeffRef(indices...);
    }

    // Resizing
    template <typename... ArgTypes>
    void conservativeResize(ArgTypes... args) {
        m_inner.conservativeResize(args...);
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    void conservativeResizeLike(const Matrix<OtherScalar,
                                             OtherRows,
                                             OtherCols,
                                             OtherOptions,
                                             OtherMaxRows,
                                             OtherMaxCols>& other) {
        m_inner.conservativeResizeLike(other.m_inner);
    }
    template <typename OtherDerived>
    void conservativeResizeLike(const Eigen::DenseBase<OtherDerived>& other) {
        m_inner.conservativeResizeLike(other);
    }
    template <typename... ArgTypes>
    void resize(ArgTypes... args) {
        m_inner.resize(args...);
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    void resizeLike(const Matrix<OtherScalar,
                                 OtherRows,
                                 OtherCols,
                                 OtherOptions,
                                 OtherMaxRows,
                                 OtherMaxCols>& other) {
        m_inner.resizeLike(other.m_inner);
    }
    template <typename OtherDerived>
    void resizeLike(const Eigen::DenseBase<OtherDerived>& other) {
        m_inner.resizeLike(other);
    }

    // Data access
    Scalar* data() { return m_inner.data(); }
    const Scalar* data() const { return m_inner.data(); }

    // Lazy assignment
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix& lazyAssign(const Matrix<OtherScalar,
                                    OtherRows,
                                    OtherCols,
                                    OtherOptions,
                                    OtherMaxRows,
                                    OtherMaxCols>& other) {
        m_inner.lazyAssign(other.m_inner);
        return *this;
    }
    template <typename OtherDerived>
    Matrix& lazyAssign(const Eigen::DenseBase<OtherDerived>& other) {
        m_inner.lazyAssign(other);
        return *this;
    }

    // Set inner values from various scalar sources
    template <typename... ArgTypes>
    Matrix& setConstant(ArgTypes&&... args) {
        m_inner.setConstant(std::forward(args)...);
        return *this;
    }
    template <typename... Index>
    Matrix& setZero(Index... indices) {
        m_inner.setZero(indices...);
        return *this;
    }
    template <typename... Index>
    Matrix& setOnes(Index... indices) {
        m_inner.setOnes(indices...);
        return *this;
    }
    template <typename... Index>
    Matrix& setRandom(Index... indices) {
        m_inner.setRandom(indices...);
        return *this;
    }

    // Map foreign data
    template <typename... ArgTypes>
    static Matrix Map(ArgTypes&&... args) {
        return Matrix(Inner::Map(std::forward(args)...));
    }
    template <typename... ArgTypes>
    static Matrix MapAligned(ArgTypes&&... args) {
        return Matrix(Inner::MapAligned(std::forward(args)...));
    }

    // TODO: Replicate interface of Eigen::MatrixBase
    // TODO: Replicate interface of Eigen::DenseBase
    // TODO: Replicate interface of all Eigen::DenseCoeffsBase types
    // TODO: Replicate interface of Eigen::EigenBase

private:
    using Inner = Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxRows>;
    Inner m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
