// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <type_traits>
#include <utility>

#include "EigenBase.hpp"
#include "EigenDense.hpp"
#include "EigenPrologue.hpp"
#include "Matrix.hpp"
#include "SolverBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Eigen::LLT wrapper
template <typename _MatrixType, int _UpLo>
class LLT : public SolverBase<LLT<_MatrixType, _UpLo>> {
    using Super = SolverBase<LLT>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using MatrixType = _MatrixType;
    static constexpr int UpLo = _UpLo;

    // Wrapped Eigen type
private:
    using MatrixTypeInner = typename MatrixType::Inner;
public:
    using Inner = Eigen::LLT<MatrixTypeInner, UpLo>;

    // Access the wrapped Eigen object
    Inner& getInner() {
        return m_inner;
    }
    const Inner& getInner() const {
        return m_inner;
    }
    Inner&& moveInner() {
        return std::move(m_inner);
    }

    // === Base class API ===

    // Re-export useful base class interface
    using Index = typename Super::Index;
    using Scalar = typename Super::Scalar;
    static constexpr int Size = Super::Rows;
    static_assert(Super::Cols == Super::Rows, "Matrix type is not square");

    // === Eigen::LLT API ===

    // Default constructor
    LLT() = default;

    // Construct factorization of a given matrix
    template <typename InputType>
    explicit LLT(const EigenBase<InputType>& matrix)
        : m_inner(matrix.derivedInner())
    {}

    // Perform in-place decomposition, if possible
    template <typename InputType>
    explicit LLT(EigenBase<InputType>& matrix)
        : m_inner(matrix.derivedInner())
    {}

    // Preallocating constructor
    explicit LLT(Index size)
        : m_inner(size)
    {}

    // Compute the Cholesky decomposition of a matrix
    template <typename InputType>
    LLT& compute(const EigenBase<InputType>& a) {
        m_inner.compute(a.derivedInner());
        return *this;
    }

    // Report whether the previous computation was successful
    ComputationInfo info() const {
        return m_inner.info();
    }

    // Access the internal LDLT decomposition matrix
    const MatrixType& matrixLLT() const {
        const auto& resultInner = m_inner.matrixLLT();
        static_assert(
            std::is_same_v<decltype(resultInner),
                           const MatrixTypeInner&>,
            "Unexpected return type from Eigen in-place accessor");
        return reinterpret_cast<const MatrixType&>(resultInner);
    }

    // Access views of the lower and upper triangular matrices L and U
    //
    // FIXME: Should provide true triangular views, not just as an optimization
    //        but also because the difference between owned values and const
    //        references is observable if the user takes a dangling reference
    //        using something like
    //        `const auto& bad = ldlt.matrixL().topLeftCorner<2, 2>()`.
    //
    using MatrixL = Matrix<Scalar, Size, Size>;
    using MatrixU = Matrix<Scalar, Size, Size>;
    MatrixL matrixL() const {
        return MatrixL(m_inner.matrixL());
    }
    MatrixU matrixU() const {
        return MatrixU(m_inner.matrixU());
    }

    // TODO: Support rankUpdate()
    //       This requires a bit of work, and isn't used by Acts yet

    // Estimate of the reciprocal condition number
private:
    using ScalarTraits = NumTraits<Scalar>;
public:
    using RealScalar = typename ScalarTraits::Real;
    RealScalar rcond() const {
        return m_inner.rcond();
    }

    // Reconstructed version of the original matrix, i.e. L x Lt*
    MatrixType reconstructedMatrix() const {
        return MatrixType(m_inner.reconstructedMatrix());
    }

private:
    Inner m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
