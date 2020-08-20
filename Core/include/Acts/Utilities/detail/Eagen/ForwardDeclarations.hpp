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

namespace Acts {

namespace detail {

namespace Eagen {

// Forward declaration of type traits (used by some other forward declarations)
template <typename EagenType>
struct TypeTraits;

// Forward declarations of concrete types
template <typename Scalar> class AngleAxis;
template <typename Derived, int BlockRows, int BlockCols, bool InnerPanel>
class Block;
template <typename MatrixType, int DiagIndex = 0> class Diagonal;
template <typename Derived> class DiagonalBase;
template <typename Scalar, int Size, int MaxSize = Size>
class DiagonalMatrix;
template <typename DiagonalVectorType> class DiagonalWrapper;
template <typename _Scalar, int _AmbientDim, int _Options = AutoAlign>
class Hyperplane;
template <typename _MatrixType,
          int _QRPreconditioner = ColPivHouseholderQRPreconditioner>
class JacobiSVD;
template <typename MatrixType, int UpLo = Lower> class LLT;
template <typename MatrixType, int UpLo = Lower> class LDLT;
template <typename Derived,
          int MapOptions = Unaligned,
          typename StrideType = Stride<TypeTraits<Derived>::OuterStride,
                                       TypeTraits<Derived>::InnerStride>>
class Map;
template <typename Scalar, int Rows, int Cols,
          int Options = AutoAlign |
                        ( (Rows==1 && Cols!=1) ? RowMajor
                          : (Cols==1 && Rows!=1) ? ColMajor
                          : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
          int MaxRows = Rows,
          int MaxCols = Cols>
class Matrix;
template <typename Derived> class MatrixBase;
template <typename _Scalar, int _AmbientDim, int _Options = AutoAlign>
class ParametrizedLine;
template <typename Derived> class PlainMatrixBase;
template <typename Scalar, int Options = AutoAlign> class Quaternion;
template <typename Scalar> class Rotation2D;
template <typename Derived, int Dim> class RotationBase;
template <typename Scalar, int Dim> class Translation;
template <typename Scalar> class UniformScaling;
template <typename Derived, int Size> class VectorBlock;

// Equivalent of Eigen's vector typedefs
template <typename Type, int Size>
using Vector = Matrix<Type, Size, 1>;
template <typename Type, int Size>
using RowVector = Matrix<Type, 1, Size>;

// We don't need to replicate Eigen's full class hierarchy for now, but let's
// keep the useful metadata from Eigen's method signatures around
template <typename Derived> using PlainObjectBase = PlainMatrixBase<Derived>;
template <typename Derived> using DenseBase = MatrixBase<Derived>;
template <typename Derived>
using DenseCoeffsBaseDirectWrite = MatrixBase<Derived>;
template <typename Derived>
using DenseCoeffsBaseWrite = MatrixBase<Derived>;
template <typename Derived>
using DenseCoeffsBaseReadOnly = MatrixBase<Derived>;

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
