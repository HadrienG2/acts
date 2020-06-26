// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <utility>

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
    using Super = DiagonalBase<DiagonalMatrix>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Scalar = _Scalar;
    static constexpr int Size = _Size;
    static constexpr int MaxSize = _MaxSize;

    // Wrapped Eigen type
    using Inner = Eigen::DiagonalMatrix<Scalar, Size, MaxSize>;

    // Access the inner Eigen object
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
    using Super::diagonal;

    // === Eigen::DiagonalMatrix API ===

    // Default constructor
    DiagonalMatrix() = default;

    // Constructor from inner Eigen type
    DiagonalMatrix(const Inner& inner)
        : m_inner(inner)
    {}

    // Constructor from other diagonal matrix
    template <typename OtherDerived>
    DiagonalMatrix(const DiagonalBase<OtherDerived>& other)
        : m_inner(other.derivedInner())
    {}

    // Constructor from vector
    template <typename OtherDerived>
    explicit DiagonalMatrix(const MatrixBase<OtherDerived>& other)
        : m_inner(other.derivedInner())
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
