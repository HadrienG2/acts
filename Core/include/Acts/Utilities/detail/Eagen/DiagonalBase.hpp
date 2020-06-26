// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <type_traits>

#include "EigenBase.hpp"
#include "ForwardDeclarations.hpp"
#include "Matrix.hpp"
#include "MatrixBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Spiritual equivalent of Eigen::DiagonalBase
template <typename _Derived>
class DiagonalBase : public EigenBase<_Derived> {
    using Super = EigenBase<_Derived>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Derived = _Derived;

    // === Base class API ===

    // Re-export useful base class interface
    using Index = typename Super::Index;
    using Super::cols;
    using Super::derived;
    using Super::derivedInner;
    using Super::rows;

    // === Eigen::DiagonalBase API ===

    // Propagate parameters of the derived class
protected:
    using DerivedTraits = typename Super::DerivedTraits;
public:
    using Scalar = typename DerivedTraits::Scalar;
    static constexpr int Size = DerivedTraits::Size;

    // Helper for rotation-diagonal products
    static constexpr bool IsVectorAtCompileTime = false;

    // Owned version of this matrix type
    using DiagonalMatrixType = DiagonalMatrix<Scalar, Size>;

    // Convert to a dense matrix
    using DenseMatrixType = Matrix<Scalar, Size, Size>;
    DenseMatrixType toDenseMatrix() const {
        return DenseMatrixType(
            derivedInner().toDenseMatrix()
        );
    }

    // Access the diagonal elements
    using DiagonalVectorType = typename DerivedTraits::DiagonalVectorType;
private:
    using DiagonalVectorTypeInner = typename DiagonalVectorType::Inner;
public:
    const DiagonalVectorType& diagonal() const {
        const auto& resultInner = derivedInner().diagonal();
        static_assert(
            std::is_same_v<decltype(resultInner),
                           const DiagonalVectorTypeInner&>,
            "Unexpected return type from Eigen in-place accessor");
        return reinterpret_cast<const DiagonalVectorType&>(resultInner);
    }
    DiagonalVectorType& diagonal() {
        auto& resultInner = derivedInner().diagonal();
        static_assert(
            std::is_same_v<decltype(resultInner),
                           DiagonalVectorTypeInner&>,
            "Unexpected return type from Eigen in-place accessor");
        return reinterpret_cast<DiagonalVectorType&>(resultInner);
    }

    // Invert this matrix
    DiagonalMatrixType inverse() const {
        return DiagonalMatrixType(
            derivedInner().inverse()
        );
    }

    // Multiply by a dense matrix
    template <typename MatrixDerived>
    Matrix<Scalar, Size, MatrixDerived::Cols>
    operator*(const MatrixBase<MatrixDerived>& matrix) const {
        return Matrix<Scalar, Size, MatrixDerived::Cols>(
            derivedInner() * matrix.derivedInner()
        );
    }

    // Multiply by a scalar on either side
    DiagonalMatrixType operator*(const Scalar& scalar) const {
        return DiagonalMatrixType(
            derivedInner() * scalar
        );
    }
    friend DiagonalMatrixType operator*(const Scalar& scalar,
                                        const DiagonalBase& other) {
        return DiagonalMatrixType(
            scalar * other.derivedInner()
        );
    }

    // Sum/difference of diagonal matrices
    template <typename OtherDerived>
    DiagonalMatrixType
    operator+(const DiagonalBase<OtherDerived>& other) const {
        return DiagonalMatrixType(
            derivedInner() + other.derivedInner()
        );
    }
    template <typename OtherDerived>
    DiagonalMatrixType
    operator-(const DiagonalBase<OtherDerived>& other) const {
        return DiagonalMatrixType(
            derivedInner() - other.derivedInner()
        );
    }
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
