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

// Eigen::Block wrapper
//
// We cannot eliminate this expression template because some code in Acts needs
// write access to matrix blocks.
//
template <typename Derived, int BlockRows, int BlockCols, bool InnerPanel>
class Block : public MatrixBase<Block<Derived,
                                      BlockRows,
                                      BlockCols,
                                      InnerPanel>> {
private:
    using DerivedTraits = TypeTraits<Derived>;
    using DerivedInner = typename DerivedTraits::Inner;
    using Inner = Eigen::Block<DerivedInner, BlockRows, BlockCols, InnerPanel>;
    using Index = Eigen::Index;

public:
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

    // Access the inner Eigen block (used for CRTP)
    Inner& getInner() {
        return m_inner;
    }
    const Inner& getInner() const {
        return m_inner;
    }
    Inner&& moveInner() {
        return std::move(m_inner);
    }

private:
    Inner m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
