// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <type_traits>

#include "EigenDense.hpp"
#include "EigenPrologue.hpp"
#include "ForwardDeclarations.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Handle const types
template <typename EagenType>
struct TypeTraits<const EagenType> : public TypeTraits<EagenType> {};

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
    template <typename _DerivedInner>
    using GenericInner = Eigen::Block<_DerivedInner,
                                      BlockRows,
                                      BlockCols,
                                      InnerPanel>;
public:
    using Inner = std::conditional_t<std::is_const_v<Derived>,
                                     GenericInner<const DerivedInner>,
                                     GenericInner<DerivedInner>>;
    using Scalar = typename DerivedTraits::Scalar;
    static constexpr int Rows = BlockRows;
    static constexpr int Cols = BlockCols;
    static constexpr int Options = DerivedTraits::Options;
    static constexpr int MaxRows = Inner::MaxRowsAtCompileTime;
    static constexpr int MaxCols = Inner::MaxColsAtCompileTime;
    static constexpr int InnerStride = DerivedTraits::InnerStride;
    static constexpr int OuterStride = DerivedTraits::OuterStride;
};

// Matrix diagonal wrapper type traits
template <typename MatrixType, int DiagIndex>
struct TypeTraits<Diagonal<MatrixType, DiagIndex>> {
private:
    using MatrixTypeTraits = TypeTraits<MatrixType>;
    using MatrixTypeInner = typename MatrixTypeTraits::Inner;
public:
    using Scalar = typename MatrixTypeInner::Scalar;
    static constexpr int Rows =
        std::min(MatrixTypeTraits::Rows - std::max(-DiagIndex, 0),
                 MatrixTypeTraits::Cols - std::max(DiagIndex, 0));
    static constexpr int Cols = 1;
    static constexpr int Options = MatrixTypeTraits::Options;
    static constexpr int MaxRows =
        std::min(MatrixTypeTraits::MaxRows - std::max(-DiagIndex, 0),
                 MatrixTypeTraits::MaxCols - std::max(DiagIndex, 0));
    static constexpr int MaxCols = 1;
    using Inner = Eigen::Diagonal<MatrixTypeInner, DiagIndex>;
};

// Diagonal matrix type traits
template <typename _Scalar, int _Size, int _MaxSize>
struct TypeTraits<DiagonalMatrix<_Scalar, _Size, _MaxSize>> {
    using Scalar = _Scalar;
    static constexpr int Size = _Size;
    static constexpr int MaxSize = _MaxSize;
    using DiagonalVectorType = Matrix<Scalar, Size, 1, 0, MaxSize, 1>;
    using Inner = Eigen::DiagonalMatrix<Scalar, Size, MaxSize>;
};

// Jacobi SVD decomposition type traits
template <typename _MatrixType, int _QRPreconditioner>
struct TypeTraits<JacobiSVD<_MatrixType, _QRPreconditioner>> {
public:
    using MatrixType = _MatrixType;
    static constexpr int QRPreconditioner = _QRPreconditioner;
private:
    using MatrixTypeInner = typename TypeTraits<MatrixType>::Inner;
public:
    using Inner = Eigen::JacobiSVD<MatrixTypeInner, QRPreconditioner>;
};

// Standard Cholesky decomposition type traits
template <typename _MatrixType, int _UpLo>
struct TypeTraits<LLT<_MatrixType, _UpLo>> {
    using MatrixType = _MatrixType;
    static constexpr int UpLo = _UpLo;
private:
    using MatrixTypeInner = typename TypeTraits<MatrixType>::Inner;
public:
    using Inner = Eigen::LLT<MatrixTypeInner, UpLo>;
};

// Robust Cholesky decomposition type traits
template <typename _MatrixType, int _UpLo>
struct TypeTraits<LDLT<_MatrixType, _UpLo>> {
    using MatrixType = _MatrixType;
    static constexpr int UpLo = _UpLo;
private:
    using MatrixTypeInner = typename TypeTraits<MatrixType>::Inner;
public:
    using Inner = Eigen::LDLT<MatrixTypeInner, UpLo>;
};

// Mapped array type traits
template <typename Derived, int MapOptions, typename StrideType>
struct TypeTraits<Map<Derived, MapOptions, StrideType>> {
private:
    using DerivedTraits = TypeTraits<Derived>;
    using DerivedInner = typename DerivedTraits::Inner;
    template <typename _DerivedInner>
    using GenericInner = Eigen::Map<_DerivedInner, MapOptions, StrideType>;
public:
    using Inner = std::conditional_t<std::is_const_v<Derived>,
                                     GenericInner<const DerivedInner>,
                                     GenericInner<DerivedInner>>;
    using Scalar = typename DerivedTraits::Scalar;
    static constexpr int Rows = Inner::RowsAtCompileTime;
    static constexpr int Cols = Inner::ColsAtCompileTime;
    static constexpr int Options = DerivedTraits::Options;
    static constexpr int MaxRows = Inner::MaxRowsAtCompileTime;
    static constexpr int MaxCols = Inner::MaxColsAtCompileTime;
    static constexpr int InnerStride = StrideType::InnerStrideAtCompileTime;
    static constexpr int OuterStride = StrideType::OuterStrideAtCompileTime;
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
    template <typename _DerivedInner>
    using GenericInner = Eigen::VectorBlock<_DerivedInner, Size>;
public:
    using Inner = std::conditional_t<std::is_const_v<Derived>,
                                     GenericInner<const DerivedInner>,
                                     GenericInner<DerivedInner>>;
    using Scalar = typename DerivedTraits::Scalar;
    static constexpr int Rows = Inner::RowsAtCompileTime;
    static constexpr int Cols = Inner::ColsAtCompileTime;
    static constexpr int Options = DerivedTraits::Options;
    static constexpr int MaxRows = Inner::MaxRowsAtCompileTime;
    static constexpr int MaxCols = Inner::MaxColsAtCompileTime;
    static constexpr int InnerStride = DerivedTraits::InnerStride;
    static constexpr int OuterStride = DerivedTraits::OuterStride;
};

// 2D rotation type traits
template <typename _Scalar>
struct TypeTraits<Rotation2D<_Scalar>> {
    using Scalar = _Scalar;
    using Inner = Eigen::Rotation2D<Scalar>;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
