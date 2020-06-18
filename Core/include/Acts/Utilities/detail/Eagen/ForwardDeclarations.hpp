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

// Forward declarations
template <typename Scalar> class AngleAxis;
template <typename Derived, int BlockRows, int BlockCols, bool InnerPanel>
class Block;
template <typename _MatrixType,
          int _QRPreconditioner = ColPivHouseholderQRPreconditioner>
class JacobiSVD;
template <typename Derived, int MapOptions, typename StrideType> class Map;
template <typename Scalar, int Rows, int Cols,
          int Options = AutoAlign |
                        ( (Rows==1 && Cols!=1) ? RowMajor
                          : (Cols==1 && Rows!=1) ? ColMajor
                          : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
          int MaxRows = Rows,
          int MaxCols = Cols>
class Matrix;
template <typename Derived> class MatrixBase;
template <typename Derived> class PlainMatrixBase;
template <typename Scalar, int Options = AutoAlign> class Quaternion;
template <typename Scalar> class Rotation2D;
template <typename Derived, int Dim> class RotationBase;
template <typename Scalar, int Dim> class Translation;
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
