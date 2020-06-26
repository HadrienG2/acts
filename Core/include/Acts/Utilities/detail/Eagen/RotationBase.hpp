// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <type_traits>

#include "EigenDense.hpp"
#include "EigenPrologue.hpp"
#include "ForwardDeclarations.hpp"
#include "Transform.hpp"
#include "TypeTraits.hpp"
#include "UniformScaling.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

namespace detail2 {
    template <typename RotationDerived,
              typename OtherDerived,
              bool OtherIsVector>
    struct RotationEigenProdResult;

    template <typename RotationDerived, typename OtherMatrix>
    struct RotationEigenProdResult<RotationDerived, OtherMatrix, false> {
    private:
        static constexpr int Dim = RotationDerived::Dim;
        using Scalar = typename RotationDerived::Scalar;
    public:
        using Type = Matrix<Scalar, Dim, Dim>;
    };

    template <typename RotationDerived, typename Scalar, int Dim, int MaxDim>
    struct RotationEigenProdResult<RotationDerived,
                                   DiagonalMatrix<Scalar, Dim, MaxDim>,
                                   false> {
        using Type = Transform<Scalar, Dim, Affine>;
    };

    template <typename RotationDerived, typename OtherVector>
    struct RotationEigenProdResult<RotationDerived, OtherVector, true> {
    private:
        static constexpr int Dim = RotationDerived::Dim;
        using Scalar = typename RotationDerived::Scalar;
    public:
        using Type = Vector<Scalar, Dim>;
    };
}

// Spiritual equivalent of Eigen::RotationBase
template <typename _Derived, int _Dim>
class RotationBase {
public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Derived = _Derived;
    static constexpr int Dim = _Dim;

    // Wrapped Eigen type
protected:
    using DerivedTraits = TypeTraits<Derived>;
public:
    using Inner = typename DerivedTraits::Inner;

    // Access the wrapped Eigen object
    Inner& derivedInner() {
        return derived().getInner();
    }
    const Inner& derivedInner() const {
        return derived().getInner();
    }
    Inner&& moveDerivedInner() {
        return derived().moveInner();
    }

    // === Eigen::RotationBase API ===

    // Scalar type
    using Scalar = typename DerivedTraits::Scalar;

    // Rotation matrix type for this rotation
    using RotationMatrixType = Matrix<Scalar, Dim, Dim>;

    // CRTP daughter class access
    Derived& derived() {
        return *static_cast<Derived*>(this);
    }
    const Derived& derived() const {
        return *static_cast<const Derived*>(this);
    }

    // Invert the rotation
    Derived inverse() const {
        return Derived(derivedInner().inverse());
    }

    // Compute the associated rotation matrix
    RotationMatrixType matrix() const {
        return RotationMatrixType(derivedInner().matrix());
    }
    RotationMatrixType toRotationMatrix() const {
        return RotationMatrixType(derivedInner().toRotationMatrix());
    }

    // Multiply by a matrix or vector
    // NOTE: No need for Eigen::EigenBase version as this does not support Array
protected:
    template <typename OtherDerived>
    using MatProdResult =
        typename detail2::RotationEigenProdResult<
            Derived, OtherDerived, OtherDerived::IsVectorAtCompileTime>::Type;
public:
    template <typename OtherDerived>
    MatProdResult<OtherDerived>
    operator*(const EigenBase<OtherDerived>& e) const {
        return MatProdResult<OtherDerived>(
            derivedInner() * e.derivedInner()
        );
    }

    // Left-hand-side multiplication by a matrix
    template<typename OtherDerived> friend
    RotationMatrixType operator*(const EigenBase<OtherDerived>& l,
                                 const Derived& r)
    {
        return RotationMatrixType(l.derivedInner() * r.getInner());
    }
    template<typename OtherDerived> friend
    RotationMatrixType operator*(const Eigen::EigenBase<OtherDerived>& l,
                                 const Derived& r)
    {
        return RotationMatrixType(l * r.getInner());
    }

    // Multiply by another transform
    template <int Mode, int Options>
    Transform<Scalar, Dim, Mode>
    operator*(const Transform<Scalar, Dim, Mode, Options>& t) const {
        return Transform<Scalar, Dim, Mode>(
            derivedInner() * t.getInner()
        );
    }

    // Compose with a translation
    Transform<Scalar, Dim, Isometry>
    operator*(const Translation<Scalar, Dim>& t) const {
        return Transform<Scalar, Dim, Isometry>(derivedInner() * t.getInner());
    }

    // Compose with a uniform scaling
    RotationMatrixType operator*(const UniformScaling<Scalar>& s) const {
        return RotationMatrixType(
            derivedInner() * s.getInner()
        );
    }
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
