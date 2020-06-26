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

#include "EigenDense.hpp"
#include "EigenPrologue.hpp"
#include "Matrix.hpp"
#include "SolverBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Eigen::JacobiSVD wrapper
template <typename _MatrixType, int _QRPreconditioner>
class JacobiSVD : public SolverBase<JacobiSVD<_MatrixType, _QRPreconditioner>> {
    using Super = SolverBase<JacobiSVD>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using MatrixType = _MatrixType;
    static constexpr int QRPreconditioner = _QRPreconditioner;

    // Wrapped Eigen type
private:
    using MatrixTypeInner = typename MatrixType::Inner;
public:
    using Inner = Eigen::JacobiSVD<MatrixTypeInner, QRPreconditioner>;

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
    static constexpr int Rows = Super::Rows;
    static constexpr int Cols = Super::Cols;
    static constexpr int Options = Super::Options;
    static constexpr int MaxRows = Super::MaxRows;
    static constexpr int MaxCols = Super::MaxCols;

    // === Eigen::SVDBase API ===

    // Real version of the scalar type
    using RealScalar = typename Inner::RealScalar;

    // Truth that the U and V matrices will be computed
    bool computeU() const {
        return m_inner.computeU();
    }
    bool computeV() const {
        return m_inner.computeV();
    }

    // Access matrices U and V
    using MatrixUType = Matrix<Scalar, Rows, Rows, Options, MaxRows, MaxRows>;
private:
    using MatrixUTypeInner = typename MatrixUType::Inner;
public:
    const MatrixUType& matrixU() const {
        const auto& resultInner = m_inner.matrixU();
        static_assert(
            std::is_same_v<decltype(resultInner),
                           const MatrixUTypeInner&>,
            "Unexpected return type from Eigen in-place accessor");
        return reinterpret_cast<const MatrixUType&>(resultInner);
    }
    using MatrixVType = Matrix<Scalar, Cols, Cols, Options, MaxCols, MaxCols>;
private:
    using MatrixVTypeInner = typename MatrixVType::Inner;
public:
    const MatrixVType& matrixV() const {
        const auto& resultInner = m_inner.matrixV();
        static_assert(
            std::is_same_v<decltype(resultInner),
                           const MatrixVTypeInner&>,
            "Unexpected return type from Eigen in-place accessor");
        return reinterpret_cast<const MatrixVType&>(resultInner);
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
    using SingularValuesType = Matrix<Scalar,
                                      std::min(Rows, Cols),
                                      1,
                                      Options & ~RowMajor,
                                      std::min(MaxRows, MaxCols),
                                      1>;
private:
    using SingularValuesTypeInner = typename SingularValuesType::Inner;
public:
    const SingularValuesType& singularValues() const {
        const auto& resultInner = m_inner.singularValues();
        static_assert(
            std::is_same_v<decltype(resultInner),
                           const SingularValuesTypeInner&>,
            "Unexpected return type from Eigen in-place accessor");
        return reinterpret_cast<const SingularValuesType&>(resultInner);
    }

    // === Eigen::JacobiSVD API ===

    // Default constructor
    JacobiSVD() = default;

    // Constructor from matrix and optionally decomposition options
    explicit JacobiSVD(const MatrixType& matrix, int computationOptions = 0)
        : m_inner(matrix.getInner(), computationOptions)
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

    // Perform the decomposition using custom options
    JacobiSVD& compute(const MatrixType& matrix,
                       unsigned int computationOptions = 0) {
        m_inner.compute(matrix.derivedInner(), computationOptions);
        return *this;
    }

private:
    Inner m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
