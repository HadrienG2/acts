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
#include "TypeTraits.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Eigen::Block wrapper
//
// We cannot eliminate this expression template because some code in Acts needs
// write access to matrix blocks.
//
template <typename _Derived, int _BlockRows, int _BlockCols, bool _InnerPanel>
class Block : public MatrixBase<Block<_Derived,
                                      _BlockRows,
                                      _BlockCols,
                                      _InnerPanel>> {
    using Super = MatrixBase<Block>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Derived = _Derived;
    static constexpr int BlockCols = _BlockCols;
    static constexpr int BlockRows = _BlockRows;
    static constexpr int InnerPanel = _InnerPanel;

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

    // === Eigen::Block API ===

    // Eigen-style constructor from an Eagen expression
    Block(Derived& xpr, Index i) : m_inner(xpr.getInner(), i) {}
    Block(Derived& xpr, Index startRow, Index startCol)
        : m_inner(xpr.getInner(), startRow, startCol)
    {}
    Block(Derived& xpr,
          Index startRow,
          Index startCol,
          Index blockRows,
          Index blockCols)
        : m_inner(xpr.getInner(), startRow, startCol, blockRows, blockCols)
    {}

    // Wrap a pre-existing Eigen block expression
    template <typename BlockXpr>
    Block(BlockXpr&& block) : m_inner(std::move(block)) {}

    // Assignment from Eagen and Eigen expressions
    //
    // NOTE: Remove Eigen::EigenBase overloads if we ever wrap Eigen::Array.
    //
    template <typename OtherDerived>
    Block& operator=(const EigenBase<OtherDerived>& other) {
        m_inner = other.derivedInner();
        return *this;
    }
    template <typename OtherDerived>
    Block& operator=(const Eigen::EigenBase<OtherDerived>& other) {
        m_inner = other;
        return *this;
    }
#if EIGEN_HAS_RVALUE_REFERENCES
    template <typename OtherDerived>
    Block& operator=(EigenBase<OtherDerived>&& other) {
        m_inner = other.moveDerivedInner();
        return *this;
    }
    template <typename OtherDerived>
    Block& operator=(Eigen::EigenBase<OtherDerived>&& other) {
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
