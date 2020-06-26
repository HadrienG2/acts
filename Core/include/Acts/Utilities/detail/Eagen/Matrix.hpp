// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <utility>
#include <type_traits>

#include "EigenDense.hpp"
#include "ForwardDeclarations.hpp"
#include "PlainMatrixBase.hpp"
#include "RotationBase.hpp"

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
    using Super = PlainMatrixBase<Matrix>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Scalar = _Scalar;
    static constexpr int Rows = _Rows;
    static constexpr int Cols = _Cols;
    static constexpr int Options = _Options;
    static constexpr int MaxRows = _MaxRows;
    static constexpr int MaxCols = _MaxCols;

    // Wrapped Eigen type
    using Inner = Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;

    // Access the wrapped Eigen object
    Inner& getInner() {
        return m_inner;
    }
    const Inner& getInner() const {
        return m_inner;
    }
    Inner&& moveInner() {
        return std::move(m_inner);
    }

    // === Base class API ===

    // Re-export useful base class interface
    using Index = typename Super::Index;
    using RealScalar = typename Super::RealScalar;

    // === Eigen::Matrix interface ===

    // Default constructor
    Matrix() = default;

    // Copy constructor and assignment from an expression
    //
    // TODO: Simplify if we add Array support to Eagen
    //
    template <typename OtherDerived>
    Matrix(const EigenBase<OtherDerived>& other)
        : m_inner(other.derivedInner())
    {}
    template <typename OtherDerived>
    Matrix(const Eigen::EigenBase<OtherDerived>& other)
        : m_inner(other)
    {}
    template <typename OtherDerived>
    Matrix& operator=(const EigenBase<OtherDerived>& other) {
        m_inner = other.derivedInner();
        return *this;
    }
    template <typename OtherDerived>
    Matrix& operator=(const Eigen::EigenBase<OtherDerived>& other) {
        m_inner = other;
        return *this;
    }

    // Move construction and assignment, if supported by Eigen
    //
    // TODO: Simplify if we add Array support to Eagen
    //
#if EIGEN_HAS_RVALUE_REFERENCES
    template <typename OtherDerived>
    Matrix(EigenBase<OtherDerived>&& other)
        : m_inner(other.moveDerivedInner())
    {}
    template <typename OtherDerived>
    Matrix(Eigen::EigenBase<OtherDerived>&& other)
        : m_inner(std::move(other))
    {}
    template <typename OtherDerived>
    Matrix& operator=(EigenBase<OtherDerived>&& other) {
        m_inner = other.moveDerivedInner();
        return *this;
    }
    template <typename OtherDerived>
    Matrix& operator=(Eigen::EigenBase<OtherDerived>&& other) {
        m_inner = std::move(other);
        return *this;
    }
#endif

    // Construction and assignment from Eigen rotation
    template <typename OtherDerived>
    Matrix(const RotationBase<OtherDerived, Cols>& r)
        : m_inner(r.derivedInner())
    {}
    template <typename OtherDerived>
    Matrix& operator=(const RotationBase<OtherDerived, Cols>& r) {
        m_inner = r.derivedInner();
        return *this;
    }

    // Construct a 1-dimensional vector from a scalar, or an uninitialized
    // dynamic-sized vector from a size (let Eigen resolve the ambiguity...)
    template <typename T>
    explicit Matrix(
        const T& t,
        std::enable_if_t<std::is_convertible_v<T, Index>
                         || std::is_convertible_v<T, Scalar>,
                         T>* = nullptr

    ) : m_inner(t) {}

    // Constuct a 2-dimensional vector from two scalars, or an uninitialized
    // dynamic-sized matrix from rows/cols (let Eigen resolve the ambiguity...)
    template <typename T,
              typename U,
              typename = std::enable_if_t<
                             (
                                 std::is_convertible_v<T, Index>
                                 && std::is_convertible_v<U, Index>
                             ) || (
                                 std::is_convertible_v<T, Scalar>
                                 && std::is_convertible_v<U, Scalar>
                             ),
                             void>>
    Matrix(const T& t, const U& u) : m_inner(t, u) {}

    // Construct a 3D+ vector from a set of scalars
    template <typename... Scalars>
    Matrix(const Scalar& x,
           const Scalar& y,
           const Scalar& z,
           const Scalars&... other)
        : m_inner(x, y, z, other...)
    {}

    // Matrix construction from a C-style array of coefficients
    explicit Matrix(const Scalar* data) : m_inner(data) {}

    // Emulate Eigen::Matrix's base class typedef
    using Base = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;

private:
    Inner m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
