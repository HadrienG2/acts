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

namespace Acts {

namespace detail {

namespace Eagen {

// Some type traits to ease manipulating incomplete types
template <typename EagenType>
struct TypeTraits;

// Matrix type traits
template <typename _Scalar,
          int _Rows,
          int _Cols,
          int _Options,
          int _MaxRows,
          int _MaxCols>
struct TypeTraits<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
    using Scalar = _Scalar;
    static constexpr int Rows = _Rows;
    static constexpr int Cols = _Cols;
    static constexpr int Options = _Options;
    static constexpr int MaxRows = _MaxRows;
    static constexpr int MaxCols = _MaxCols;
    using Inner = Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
};

// Sub-matrix type traits
template <typename Derived, int BlockRows, int BlockCols, bool InnerPanel>
struct TypeTraits<Block<Derived, BlockRows, BlockCols, InnerPanel>> {
private:
    using DerivedTraits = TypeTraits<Derived>;
    using DerivedInner = typename DerivedTraits::Inner;
public:
    using Scalar = typename DerivedTraits::Scalar;
    static constexpr int Rows = BlockRows;
    static constexpr int Cols = BlockCols;
    static constexpr int Options = DerivedTraits::Options;
    static constexpr int MaxRows = DerivedInner::MaxRowsAtCompileTime;
    static constexpr int MaxCols = DerivedInner::MaxColsAtCompileTime;
    using Inner = Eigen::Block<DerivedInner, BlockRows, BlockCols, InnerPanel>;
};

// Sub-vector type traits
template <typename Derived, int Size>
struct TypeTraits<VectorBlock<Derived, Size>> {
private:
    using DerivedTraits = TypeTraits<Derived>;
    using DerivedInner = typename DerivedTraits::Inner;
public:
    using Scalar = typename DerivedTraits::Scalar;
    static constexpr int Rows = DerivedInner::RowsAtCompileTime;
    static constexpr int Cols = DerivedInner::ColsAtCompileTime;
    static constexpr int Options = DerivedTraits::Options;
    static constexpr int MaxRows = DerivedInner::MaxRowsAtCompileTime;
    static constexpr int MaxCols = DerivedInner::MaxColsAtCompileTime;
    using Inner = Eigen::VectorBlock<DerivedInner, Size>;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
