// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "DiagonalBase.hpp"
#include "EigenDense.hpp"
#include "MatrixBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Wrapper of Eigen::DiagonalMatrix
template <typename _Scalar, int _Size, int _MaxSize>
class DiagonalMatrix : public DiagonalBase<DiagonalMatrix<_Scalar,
                                                          _Size,
                                                          _MaxSize>> {
private:
    // Superclass
    using Super = DiagonalBase<DiagonalMatrix<_Scalar, _Size, _MaxSize>>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Scalar = _Scalar;
    static constexpr int Size = _Size;
    static constexpr int MaxSize = _MaxSize;

    // Underlying Eigen matrix type
    using Inner = Eigen::DiagonalMatrix<Scalar, Size, MaxSize>;

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

    // === Eigen::DiagonalMatrix API ===

    // Inherit useful methods from superclass
    using Super::diagonal;

    // Default constructor
    DiagonalMatrix() = default;

    // Constructor from other diagonal matrix
    template <typename OtherDerived>
    explicit DiagonalMatrix(const DiagonalBase<OtherDerived>& other)
        : m_inner(other.derivedInner())
    {}
    template <typename OtherDerived>
    explicit DiagonalMatrix(const Eigen::DiagonalBase<OtherDerived>& other)
        : m_inner(other)
    {}

    // Constructor from vector
    template <typename OtherDerived>
    explicit DiagonalMatrix(const MatrixBase<OtherDerived>& other)
        : m_inner(other.derivedInner())
    {}
    template <typename OtherDerived>
    explicit DiagonalMatrix(const Eigen::MatrixBase<OtherDerived>& other)
        : m_inner(other)
    {}

    // Constructor from a set of scalars
    template <typename... Scalars>
    DiagonalMatrix(const Scalar& x, const Scalar& y, const Scalars&... other)
        : m_inner(x, y, other...)
    {}

    // Uninitialized constructor from dimension
    explicit DiagonalMatrix(Index size)
        : m_inner(size)
    {}

    // Assignment from other diagonal matrix
    template <typename OtherDerived>
    DiagonalMatrix& operator=(const DiagonalBase<OtherDerived>& other) {
        m_inner = other.derivedInner();
        return *this;
    }
    template <typename OtherDerived>
    DiagonalMatrix& operator=(const Eigen::DiagonalBase<OtherDerived>& other) {
        m_inner = other.derivedInner();
        return *this;
    }

    // Resize a dynamic diagonal matrix
    void resize(Index size) {
        m_inner.resize(size);
    }

    // Set to the identity matrix
    void setIdentity() {
        m_inner.setIdentity();
    }
    void setIdentity(Index size) {
        m_inner.setIdentity(size);
    }

    // Set to zero
    void setZero() {
        m_inner.setZero();
    }
    void setZero(Index size) {
        m_inner.setZero(size);
    }

private:
    Inner m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
