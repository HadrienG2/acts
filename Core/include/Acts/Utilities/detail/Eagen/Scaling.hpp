// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <complex>

#include "DiagonalMatrix.hpp"
#include "MatrixBase.hpp"
#include "UniformScaling.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Construct a uniform scaling from a scale factor
inline UniformScaling<float> Scaling(float s) {
    return UniformScaling<float>(s);
}
inline UniformScaling<double> Scaling(double s) {
    return UniformScaling<double>(s);
}
template <typename RealScalar>
inline UniformScaling<std::complex<RealScalar>>
Scaling(const std::complex<RealScalar>& s) {
    return UniformScaling<std::complex<RealScalar>>(s);
}

// Construct an N-dimensional axis aligned scaling
template <typename Scalar, typename... Scalars>
inline DiagonalMatrix<Scalar, 2+sizeof...(Scalars)>
Scaling(const Scalar& sx, const Scalar& sy, const Scalars&... others) {
    return DiagonalMatrix<Scalar, 2+sizeof...(Scalars)>(
        sx, sy, others...
    );
}

// Constuct an axis aligned scaling from a vector expression
template <typename Derived>
inline DiagonalMatrix<typename Derived::Scalar,
                      std::max(Derived::Rows, Derived::Cols)>
Scaling(const MatrixBase<Derived>& coeffs) {
    return coeffs.asDiagonal();
}

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts