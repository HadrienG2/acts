// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EigenBase.hpp"
#include "EigenDense.hpp"
#include "Matrix.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Spiritual equivalent of Eigen::SolverBase
template <typename Derived>
class SolverBase : public EigenBase<Derived> {
private:
    // Access the parent class and derived class type traits
    using Super = EigenBase<Derived>;
    using DerivedTraits = typename Super::DerivedTraits;

public:
    // Bring back some of the parent interface which for some reason gets lost
    using Super::derived;
    using Super::derivedInner;

    // Expose decomposed matrix type information
    using MatrixType = typename DerivedTraits::MatrixType;
    using Scalar = typename MatrixType::Scalar;
    static constexpr int Rows = MatrixType::Rows;
    static constexpr int Cols = MatrixType::Cols;

    // TODO: Support transpose() and adjoint()
    //       These require quite a bit of work and aren't used by Acts yet

    // Resolve linear equation Ax = b using this SVD decomposition of A
    template <typename Rhs>
    Vector<Scalar, Cols> solve(const MatrixBase<Rhs>& b) const {
        return Vector<Scalar, Cols>(
            derivedInner().solve(b.derivedInner())
        );
    }
    template <typename Rhs>
    Vector<Scalar, Cols> solve(const Eigen::MatrixBase<Rhs>& b) const {
        return Vector<Scalar, Cols>(
            derivedInner().solve(b)
        );
    }
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
