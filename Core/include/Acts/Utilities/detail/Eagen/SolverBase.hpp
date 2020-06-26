// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EigenBase.hpp"
#include "ForwardDeclarations.hpp"
#include "Matrix.hpp"
#include "MatrixBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Spiritual equivalent of Eigen::SolverBase
template <typename _Derived>
class SolverBase : public EigenBase<_Derived> {
    using Super = EigenBase<_Derived>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Derived = _Derived;

    // === Base class API ===

    // Re-export useful base class interface
    using Index = typename Super::Index;
    using Super::derived;
    using Super::derivedInner;

    // === Eigen::SolverBase API ===

    // Expose decomposed matrix type information
protected:
    using DerivedTraits = typename Super::DerivedTraits;
public:
    using MatrixType = typename DerivedTraits::MatrixType;
    using Scalar = typename MatrixType::Scalar;
    static constexpr int Rows = MatrixType::Rows;
    static constexpr int Cols = MatrixType::Cols;
    static constexpr int Options = MatrixType::Options;
    static constexpr int MaxRows = MatrixType::MaxRows;
    static constexpr int MaxCols = MatrixType::MaxCols;

    // TODO: Support transpose() and adjoint()
    //       These require quite a bit of work and aren't used by Acts yet

    // Resolve linear equation Ax = b using this SVD decomposition of A
    template <typename Rhs>
    Vector<Scalar, Cols> solve(const MatrixBase<Rhs>& b) const {
        return Vector<Scalar, Cols>(
            derivedInner().solve(b.derivedInner())
        );
    }
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
