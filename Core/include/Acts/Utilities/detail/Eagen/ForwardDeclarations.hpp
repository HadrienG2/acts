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
template <typename Scalar, int Rows, int Cols,
          int Options = AutoAlign |
                        ( (Rows==1 && Cols!=1) ? RowMajor
                          : (Cols==1 && Rows!=1) ? ColMajor
                          : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
          int MaxRows = Rows,
          int MaxCols = Cols>
class Matrix;
template <typename Derived> class PlainObjectBase;

// We don't need to replicate Eigen's full class hierarchy for now, but let's
// keep the useful metadata from Eigen's method signatures around
template <typename Derived> using DenseBase = PlainObjectBase<Derived>;
template <typename Derived>
using DenseCoeffsBaseDirectWrite = PlainObjectBase<Derived>;
template <typename Derived>
using DenseCoeffsBaseWrite = PlainObjectBase<Derived>;
template <typename Derived>
using DenseCoeffsBaseReadOnly = PlainObjectBase<Derived>;
template <typename Derived> using EigenBase = PlainObjectBase<Derived>;

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
