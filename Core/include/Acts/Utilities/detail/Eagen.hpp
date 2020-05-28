// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// for GNU: ignore this specific warning, otherwise just include Eigen/Dense
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include <Eigen/Dense>
#pragma GCC diagnostic pop
#else
#include <Eigen/Dense>
#endif

namespace Acts {

namespace detail {

/// Eagerly evaluated variant of Eigen, without expression templates
namespace Eagen {

template <typename _Scalar, int _Rows, int _Cols,
          int _Options = Eigen::AutoAlign |
                         ( (_Rows==1 && _Cols!=1) ? Eigen::RowMajor
                           : (_Cols==1 && _Rows!=1) ? Eigen::ColMajor
                           : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
          int _MaxRows = _Rows,
          int _MaxCols = _Cols>
class Matrix {
public:
    // TODO: Replicate interface of Eigen::Matrix
    // TODO: Replicate interface of Eigen::PlainObjectBase
    // TODO: Replicate interface of Eigen::DenseBase
    // TODO: Replicate interface of all Eigen::DenseCoeffsBase types
    // TODO: Replicate interface of Eigen::EigenBase

private:
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxRows> m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
