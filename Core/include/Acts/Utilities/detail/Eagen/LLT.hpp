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
#include "SolverBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Eigen::LLT wrapper
template <typename _MatrixType, int _UpLo>
class LLT : public SolverBase<LLT<_MatrixType, _UpLo>> {
public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using MatrixType = _MatrixType;
    static constexpr int UpLo = _UpLo;

    // Eigen type that is being wrapped
private:
    using MatrixTypeInner = typename MatrixType::Inner;
public:
    using Inner = Eigen::LLT<MatrixTypeInner, UpLo>;

    // Access the inner Eigen matrix (used for CRTP)
    Inner& getInner() {
        return m_inner;
    }
    const Inner& getInner() const {
        return m_inner;
    }
    Inner&& moveInner() {
        return std::move(m_inner);
    }

    // Bring back useful superclass API
private:
    using Super = SolverBase<LLT<MatrixType, UpLo>>;
public:
    using Scalar = typename Super::Scalar;
    static constexpr int Size = Super::Rows;
    static_assert(Super::Cols == Super::Rows, "Matrix type is not square");

    // === Eigen::LLT API ===

    // Traditional Index typedef
    using Index = Eigen::Index;

    // Default constructor
    LLT() = default;

    // Construct factorization of a given matrix
    template <typename InputType>
    LLT(const EigenBase<InputType>& matrix)
        : m_inner(matrix.derivedInner())
    {}
    template <typename InputType>
    LLT(const Eigen::EigenBase<InputType>& matrix)
        : m_inner(matrix)
    {}

    // Preallocating constructor
    LLT(Index size)
        : m_inner(size)
    {}

    // Compute the Cholesky decomposition of a matrix
    template <typename InputType>
    LLT& compute(const EigenBase<InputType>& a) {
        m_inner.compute(a.derivedInner());
        return *this;
    }
    template <typename InputType>
    LLT& compute(const Eigen::EigenBase<InputType>& a) {
        m_inner.compute(a);
        return *this;
    }

    // Report whether the previous computation was successful
    ComputationInfo info() const {
        return m_inner.info();
    }

    // Access inner matrices
    // FIXME: Consider providing triangular views as a possible optimization
    using MatrixL = Matrix<Scalar, Size, Size>;
    using MatrixU = Matrix<Scalar, Size, Size>;
    MatrixL matrixL() const {
        return MatrixL(m_inner.matrixL());
    }
    MatrixType matrixLLT() const {
        return MatrixType(m_inner.matrixLLT());
    }
    MatrixU matrixU() const {
        return MatrixU(m_inner.matrixU());
    }

    // TODO: Support rankUpdate()
    //       This requires a bit of work, and isn't used by Acts yet

    // Estimate of the reciprocal condition number
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
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
