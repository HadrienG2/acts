// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <utility>

#include "EigenDense.hpp"
#include "EigenPrologue.hpp"
#include "MatrixBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Eigen::VectorBlock wrapper
//
// We cannot eliminate this expression template because some code in Acts needs
// write access to matrix blocks.
//
template <typename _Derived, int _Size>
class VectorBlock : public MatrixBase<VectorBlock<_Derived, _Size>> {
    using Super = MatrixBase<VectorBlock>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Derived = _Derived;
    static constexpr int Size = _Size;

    // Wrapped Eigen type
    using Inner = typename Super::Inner;

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

    // === Eigen::VectorBlock API ===

    // Eigen-style constructor from an Eagen expression
    VectorBlock(Derived& xpr, Index start) : m_inner(xpr.getInner(), start) {}
    VectorBlock(Derived& xpr, Index start, Index size)
        : m_inner(xpr.getInner(), start, size)
    {}

    // Assignment from Eagen and Eigen expressions
    template <typename OtherDerived>
    VectorBlock& operator=(const EigenBase<OtherDerived>& other) {
        m_inner = other.derivedInner();
        return *this;
    }
    template <typename OtherDerived>
    VectorBlock& operator=(const Eigen::EigenBase<OtherDerived>& other) {
        m_inner = other;
        return *this;
    }
#if EIGEN_HAS_RVALUE_REFERENCES
    template <typename OtherDerived>
    VectorBlock& operator=(EigenBase<OtherDerived>&& other) {
        m_inner = other.moveDerivedInner();
        return *this;
    }
    template <typename OtherDerived>
    VectorBlock& operator=(Eigen::EigenBase<OtherDerived>&& other) {
        m_inner = std::move(other);
        return *this;
    }
#endif

private:
    Inner m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
