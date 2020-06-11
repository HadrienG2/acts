// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

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
    // Eigen-compatible constructors
    Block(Derived& xpr, Index i) : m_inner(xpr.getInner(), i) {}
    Block(DerivedInner& xpr, Index i) : m_inner(xpr, i) {}
    Block(Derived& xpr, Index startRow, Index startCol)
        : m_inner(xpr.getInner(), startRow, startCol)
    {}
    Block(DerivedInner& xpr, Index startRow, Index startCol)
        : m_inner(xpr, startRow, startCol)
    {}
    Block(Derived& xpr,
          Index startRow,
          Index startCol,
          Index blockRows,
          Index blockCols)
        : m_inner(xpr.getInner(), startRow, startCol, blockRows, blockCols)
    {}
    Block(DerivedInner& xpr,
          Index startRow,
          Index startCol,
          Index blockRows,
          Index blockCols)
        : m_inner(xpr, startRow, startCol, blockRows, blockCols)
    {}

    // Wrap a pre-existing Eigen block expression
    Block(Inner&& inner) : m_inner(std::move(inner)) {}

private:
    Inner m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
