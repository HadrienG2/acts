// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EigenDense.hpp"
#include "ForwardDeclarations.hpp"
#include "PlainMatrixBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Eigen::Matrix, but with eagerly evaluated operations
template <typename _Scalar,
          int _Rows,
          int _Cols,
          int _Options,
          int _MaxRows,
          int _MaxCols>
class Matrix : public PlainMatrixBase<Matrix<_Scalar,
                                             _Rows,
                                             _Cols,
                                             _Options,
                                             _MaxRows,
                                             _MaxCols>> {
public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Scalar = _Scalar;
    static constexpr int Rows = _Rows;
    static constexpr int Cols = _Cols;
    static constexpr int Options = _Options;
    static constexpr int MaxRows = _MaxRows;
    static constexpr int MaxCols = _MaxCols;

    // Underlying Eigen matrix type (used for CRTP)
    using Inner = Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;

    // Access the inner Eigen matrix (used for CRTP)
    Inner& getInner() {
        return m_inner;
    }
    const Inner& getInner() const {
        return m_inner;
    }
    Inner&& moveInner() {
        return std::move(m_inner);
    }

    // Re-expose typedefs from Eigen
    using Index = Eigen::Index;
    using RealScalar = typename Inner::RealScalar;

    // === Eigen::Matrix interface ===

    // Basic lifecycle
    Matrix() = default;
    template <typename OtherDerived>
    Matrix(const EigenBase<OtherDerived>& other)
        : m_inner(other.derivedInner())
    {}
    template <typename OtherDerived>
    Matrix& operator=(const EigenBase<OtherDerived>& other) {
        m_inner = other.derivedInner();
        return *this;
    }
#if EIGEN_HAS_RVALUE_REFERENCES
    template <typename OtherDerived>
    Matrix(EigenBase<OtherDerived>&& other)
        : m_inner(other.moveDerivedInner())
    {}
    template <typename OtherDerived>
    Matrix& operator=(EigenBase<OtherDerived>&& other) {
        m_inner = other.moveDerivedInner();
        return *this;
    }
#endif

    // Build and assign from anything Eigen supports building or assigning from
    template <typename... Args>
    Matrix(Args&&... args) : m_inner(std::forward<Args>(args)...) {}
    template <typename Other>
    Matrix& operator=(Other&& other) {
        m_inner = std::forward<Other>(other);
        return *this;
    }

    // Emulate Eigen::Matrix's base class typedef
    using Base = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;

private:
    Inner m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
