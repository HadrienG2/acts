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
#include "ForwardDeclarations.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Some type traits to ease manipulating incomplete types
template <typename EagenType>
struct TypeTraits;

// Angle-axis transform type traits
template <typename _Scalar>
struct TypeTraits<AngleAxis<_Scalar>> {
    using Scalar = _Scalar;
    using Inner = Eigen::AngleAxis<Scalar>;
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
    static constexpr int InnerStride = DerivedTraits::InnerStride;
    static constexpr int OuterStride = DerivedTraits::OuterStride;
    using Inner = Eigen::Block<DerivedInner, BlockRows, BlockCols, InnerPanel>;
};

// Mapped array type traits
template <typename Derived, int MapOptions, typename StrideType>
struct TypeTraits<Map<Derived, MapOptions, StrideType>> {
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
    static constexpr int InnerStride = StrideType::InnerStrideAtCompileTime;
    static constexpr int OuterStride = StrideType::OuterStrideAtCompileTime;
    using Inner = Eigen::Map<DerivedInner, MapOptions, StrideType>;
};

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
    static constexpr int InnerStride = 1;
    static constexpr int OuterStride = (Options & RowMajor) ? Cols : Rows;
    using Inner = Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
};

// Quaternion type traits
template <typename _Scalar, int _Options>
struct TypeTraits<Quaternion<_Scalar, _Options>> {
    using Scalar = _Scalar;
    static constexpr int Options = _Options;
    using Inner = Eigen::Quaternion<Scalar, Options>;
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
    static constexpr int InnerStride = DerivedTraits::InnerStride;
    static constexpr int OuterStride = DerivedTraits::OuterStride;
    using Inner = Eigen::VectorBlock<DerivedInner, Size>;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
