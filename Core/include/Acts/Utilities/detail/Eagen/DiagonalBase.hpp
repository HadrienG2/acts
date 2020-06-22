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
#include "Map.hpp"
#include "Matrix.hpp"
#include "MatrixBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Spiritual equivalent of Eigen::DiagonalBase
template <typename Derived>
class DiagonalBase : public EigenBase<Derived> {
private:
    // Superclass
    using Super = EigenBase<Derived>;

protected:
    // Eigen type wrapped by the CRTP daughter class
    using DerivedTraits = typename Super::DerivedTraits;

public:
    // Re-export useful base class interface
    using Super::cols;
    using Super::derived;
    using Super::derivedInner;
    using Super::rows;

    // === Eigen::DiagonalBase API ===

    // Propagate parameters of the derived class
    using Scalar = typename DerivedTraits::Scalar;
    static constexpr int Size = DerivedTraits::Size;

    // Owned version of this matrix type
    using DiagonalMatrixType = DiagonalMatrix<Scalar, Size>;

    // Vector type which could hold the diagonal elements
    using DiagonalVectorType = Vector<Scalar, Size>;

    // Convert to a dense matrix
    using DenseMatrixType = Matrix<Scalar, Size, Size>;
    DenseMatrixType toDenseMatrix() const {
        return DenseMatrixType(
            derivedInner().toDenseMatrix()
        );
    }

    // Access the diagonal elements
    DiagonalVectorType diagonal() const {
        return DiagonalVectorType(
            derivedInner().diagonal()
        );
    }
private:
    using DiagonalVectorTypeMap = Map<DiagonalVectorType>;
public:
    DiagonalVectorTypeMap diagonal() {
        return DiagonalVectorTypeMap(
            derivedInner().diagonal().data()
        );
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
