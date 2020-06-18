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
#include "MatrixBase.hpp"
#include "Matrix.hpp"
#include "SolverBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Eigen::JacobiSVD wrapper
template <typename _MatrixType, int _QRPreconditioner>
class JacobiSVD : public SolverBase<JacobiSVD<_MatrixType, _QRPreconditioner>> {
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using MatrixType = _MatrixType;
    static constexpr int QRPreconditioner = _QRPreconditioner;

    // Eigen type that is being wrapped
    using Inner = Eigen::JacobiSVD<MatrixType, QRPreconditioner>;

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
    using Super = SolverBase<JacobiSVD<_MatrixType, _QRPreconditioner>>;
public:
    using Scalar = typename Super::Scalar;
    static constexpr int Rows = Super::Rows;
    static constexpr int Cols = Super::Cols;

    // === Eigen::SVDBase API ===

    // Eigen-style typedefs
    using Index = Eigen::Index;
    using RealScalar = typename Inner::RealScalar;

    // Truth that the U and V matrices will be computed
    bool computeU() const {
        return m_inner.computeU();
    }
    bool computeV() const {
        return m_inner.computeV();
    }

    // Access matrices U and V
    using MatrixUType = Matrix<Scalar, Rows, Rows>;
    MatrixUType matrixU() const {
        return MatrixUType(m_inner.matrixU());
    }
    using MatrixVType = Matrix<Scalar, Cols, Cols>;
    MatrixUType matrixV() const {
        return MatrixVType(m_inner.matrixV());
    }

    // Number of nonzero singular values
    Index nonzeroSingularValues() const {
        return m_inner.nonzeroSingularValues();
    }

    // Rank of the matrix that was decomposed
    Index rank() const {
        return m_inner.rank();
    }

    // Adjust threshold for what is considered nonzero in above methods
    JacobiSVD& setThreshold(const RealScalar& threshold) {
        m_inner.setThreshold(threshold);
        return *this;
    }

    // Go back to default threshold (pass special Default value here)
    JacobiSVD& setThreshold(Default_t) {
        m_inner.setThreshold(Default);
        return *this;
    }

    // Query the zero-ness threshold
    RealScalar threshold() const {
        return m_inner.threshold();
    }

    // Query the singular values
    using SingularValuesType = Vector<Scalar, std::min(Rows, Cols)>;
    SingularValuesType singularValues() const {
        return SingularValuesType(m_inner.singularValues());
    }

    // === Eigen::JacobiSVD API ===

private:
    using MatrixTypeInner = typename MatrixType::Inner;
public:

    // Default constructor
    JacobiSVD() = default;

    // Constructor from matrix and optionally decomposition options
    JacobiSVD(const MatrixType& matrix, int computationOptions = 0)
        : m_inner(matrix.getInner(), computationOptions)
    {}
    JacobiSVD(const MatrixTypeInner& matrix,
              unsigned int computationOptions = 0)
        : m_inner(matrix, computationOptions)
    {}

    // Constructor with matrix size and decomposition options
    JacobiSVD(Index rows,
              Index cols,
              unsigned int computationOptions = 0)
        : m_inner(rows, cols, computationOptions)
    {}

    // Perform the decomposition using current options
    JacobiSVD& compute(const MatrixType& matrix) {
        m_inner.compute(matrix.derivedInner());
        return *this;
    }
    JacobiSVD& compute(const MatrixTypeInner& matrix) {
        m_inner.compute(matrix);
        return *this;
    }

    // Perform the decomposition using custom options
    JacobiSVD& compute(const MatrixType& matrix,
                       unsigned int computationOptions = 0) {
        m_inner.compute(matrix.derivedInner(), computationOptions);
        return *this;
    }
    JacobiSVD& compute(const MatrixTypeInner& matrix,
                       unsigned int computationOptions = 0) {
        m_inner.compute(matrix, computationOptions);
        return *this;
    }

private:
    Inner m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
