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
struct TypeTraits<const EagenType> : TypeTraits<EagenType> {};

// Angle-axis transform type traits
template <typename _Scalar>
struct TypeTraits<AngleAxis<_Scalar>> {
    using Scalar = _Scalar;
    using Inner = Eigen::AngleAxis<Scalar>;
};

// Sub-matrix type traits
template <typename _Derived, int _BlockRows, int _BlockCols, bool _InnerPanel>
struct TypeTraits<Block<_Derived, _BlockRows, _BlockCols, _InnerPanel>> {
    using Derived = _Derived;
    static constexpr int BlockRows = _BlockRows;
    static constexpr int BlockCols = _BlockCols;
    static constexpr bool InnerPanel = _InnerPanel;
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
    using Scalar = typename Inner::Scalar;
    static constexpr int Rows = Inner::RowsAtCompileTime;
    static constexpr int Cols = Inner::ColsAtCompileTime;
    static constexpr int Options = DerivedTraits::Options;
    static constexpr int MaxRows = Inner::MaxRowsAtCompileTime;
    static constexpr int MaxCols = Inner::MaxColsAtCompileTime;
    static constexpr int InnerStride = Inner::InnerStrideAtCompileTime;
    static constexpr int OuterStride = Inner::OuterStrideAtCompileTime;
};

// Matrix diagonal wrapper type traits
template <typename _MatrixType, int _DiagIndex>
struct TypeTraits<Diagonal<_MatrixType, _DiagIndex>> {
    using MatrixType = _MatrixType;
    static constexpr int DiagIndex = _DiagIndex;
private:
    using MatrixTypeTraits = TypeTraits<MatrixType>;
    using MatrixTypeInner = typename MatrixTypeTraits::Inner;
public:
    using Inner = Eigen::Diagonal<MatrixTypeInner, DiagIndex>;
    using Scalar = typename Inner::Scalar;
    static constexpr int Rows = Inner::RowsAtCompileTime;
    static constexpr int Cols = 1;
    static constexpr int Options = MatrixTypeTraits::Options;
    static constexpr int MaxRows = Inner::ColsAtCompileTime;
    static constexpr int MaxCols = 1;
    static constexpr int InnerStride = Inner::InnerStrideAtCompileTime;
    static constexpr int OuterStride = Inner::OuterStrideAtCompileTime;
};

// Diagonal matrix type traits
template <typename _Scalar, int _Size, int _MaxSize>
struct TypeTraits<DiagonalMatrix<_Scalar, _Size, _MaxSize>>
    : TypeTraits<Matrix<_Scalar, _Size, _Size, 0, _MaxSize, _MaxSize>>
{
    using Scalar = _Scalar;
    static constexpr int Size = _Size;
    static constexpr int MaxSize = _MaxSize;
    using Inner = Eigen::DiagonalMatrix<Scalar, Size, MaxSize>;
    using DiagonalVectorType = Matrix<Scalar, Size, 1, 0, MaxSize, 1>;
};

// Diagonal matrix vector wrapper type traits
template <typename _DiagonalVectorType>
struct TypeTraits<DiagonalWrapper<_DiagonalVectorType>> {
    using DiagonalVectorType = _DiagonalVectorType;
private:
    using DiagonalVectorTraits = TypeTraits<DiagonalVectorType>;
    using DiagonalVectorTypeInner = typename DiagonalVectorTraits::Inner;
    template <typename _DiagonalVectorTypeInner>
    using GenericInner = Eigen::DiagonalWrapper<_DiagonalVectorTypeInner>;
public:
    using Inner =
        std::conditional_t<std::is_const_v<DiagonalVectorType>,
                           GenericInner<const DiagonalVectorTypeInner>,
                           GenericInner<DiagonalVectorTypeInner>>;
    using Scalar = typename Inner::Scalar;
    static constexpr int Size = std::max(DiagonalVectorTraits::Rows,
                                         DiagonalVectorTraits::Cols);
    static constexpr int Rows = Size;
    static constexpr int Cols = Size;
    static constexpr int MaxSize = std::max(DiagonalVectorTraits::MaxRows,
                                            DiagonalVectorTraits::MaxCols);
    static constexpr int MaxRows = MaxSize;
    static constexpr int MaxCols = MaxSize;
};

// Jacobi SVD decomposition type traits
template <typename _MatrixType, int _QRPreconditioner>
struct TypeTraits<JacobiSVD<_MatrixType, _QRPreconditioner>>
    : TypeTraits<_MatrixType>
{
    using MatrixType = _MatrixType;
    static constexpr int QRPreconditioner = _QRPreconditioner;
private:
    using MatrixTypeInner = typename TypeTraits<MatrixType>::Inner;
public:
    using Inner = Eigen::JacobiSVD<MatrixTypeInner, QRPreconditioner>;
};

// Standard Cholesky decomposition type traits
template <typename _MatrixType, int _UpLo>
struct TypeTraits<LLT<_MatrixType, _UpLo>> : TypeTraits<_MatrixType> {
    using MatrixType = _MatrixType;
    static constexpr int UpLo = _UpLo;
private:
    using MatrixTypeInner = typename TypeTraits<MatrixType>::Inner;
public:
    using Inner = Eigen::LLT<MatrixTypeInner, UpLo>;
};

// Robust Cholesky decomposition type traits
template <typename _MatrixType, int _UpLo>
struct TypeTraits<LDLT<_MatrixType, _UpLo>> : TypeTraits<_MatrixType> {
    using MatrixType = _MatrixType;
    static constexpr int UpLo = _UpLo;
private:
    using MatrixTypeInner = typename TypeTraits<MatrixType>::Inner;
public:
    using Inner = Eigen::LDLT<MatrixTypeInner, UpLo>;
};

// Mapped array type traits
template <typename _Derived, int _MapOptions, typename _StrideType>
struct TypeTraits<Map<_Derived, _MapOptions, _StrideType>> {
    using Derived = _Derived;
    static constexpr int MapOptions = _MapOptions;
    using StrideType = _StrideType;
private:
    using DerivedTraits = TypeTraits<Derived>;
    using DerivedInner = typename DerivedTraits::Inner;
    template <typename _DerivedInner>
    using GenericInner = Eigen::Map<_DerivedInner, MapOptions, StrideType>;
public:
    using Inner = std::conditional_t<std::is_const_v<Derived>,
                                     GenericInner<const DerivedInner>,
                                     GenericInner<DerivedInner>>;
    using Scalar = typename Inner::Scalar;
    static constexpr int Rows = Inner::RowsAtCompileTime;
    static constexpr int Cols = Inner::ColsAtCompileTime;
    static constexpr int Options = DerivedTraits::Options;
    static constexpr int MaxRows = Inner::MaxRowsAtCompileTime;
    static constexpr int MaxCols = Inner::MaxColsAtCompileTime;
    static constexpr int InnerStride = Inner::InnerStrideAtCompileTime;
    static constexpr int OuterStride = Inner::OuterStrideAtCompileTime;
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
    using Inner = Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
    static constexpr int InnerStride = Inner::InnerStrideAtCompileTime;
    static constexpr int OuterStride = Inner::OuterStrideAtCompileTime;
};

// Quaternion type traits
template <typename _Scalar, int _Options>
struct TypeTraits<Quaternion<_Scalar, _Options>> {
    using Scalar = _Scalar;
    static constexpr int Options = _Options;
    using Inner = Eigen::Quaternion<Scalar, Options>;
};

// 2D rotation type traits
template <typename _Scalar>
struct TypeTraits<Rotation2D<_Scalar>> {
    using Scalar = _Scalar;
    using Inner = Eigen::Rotation2D<Scalar>;
};

// Sub-vector type traits
template <typename _Derived, int _Size>
struct TypeTraits<VectorBlock<_Derived, _Size>> {
    using Derived = _Derived;
    static constexpr int Size = _Size;
private:
    using DerivedTraits = TypeTraits<Derived>;
    using DerivedInner = typename DerivedTraits::Inner;
    template <typename _DerivedInner>
    using GenericInner = Eigen::VectorBlock<_DerivedInner, Size>;
public:
    using Inner = std::conditional_t<std::is_const_v<Derived>,
                                     GenericInner<const DerivedInner>,
                                     GenericInner<DerivedInner>>;
    using Scalar = typename Inner::Scalar;
    static constexpr int Rows = Inner::RowsAtCompileTime;
    static constexpr int Cols = Inner::ColsAtCompileTime;
    static constexpr int Options = DerivedTraits::Options;
    static constexpr int MaxRows = Inner::MaxRowsAtCompileTime;
    static constexpr int MaxCols = Inner::MaxColsAtCompileTime;
    static constexpr int InnerStride = Inner::InnerStrideAtCompileTime;
    static constexpr int OuterStride = Inner::OuterStrideAtCompileTime;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
