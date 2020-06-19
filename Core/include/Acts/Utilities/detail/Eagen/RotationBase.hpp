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

// Spiritual equivalent of Eigen::RotationBase
template <typename _Derived, int _Dim>
class RotationBase {
public:
    // Re-expose template parameters
    using Derived = _Derived;
    static constexpr int Dim = _Dim;

protected:
    // Eigen type wrapped by the CRTP daughter class
    using DerivedTraits = TypeTraits<Derived>;
    using Inner = typename DerivedTraits::Inner;

public:
    // === Eigen::RotationBase API ===

    // Typedefs
    using Scalar = typename DerivedTraits::Scalar;
    using RotationMatrixType = Matrix<Scalar, Dim, Dim>;

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
protected:
    template <typename OtherDerived>
    using MatProdResult =
        std::conditional_t<OtherDerived::IsVectorAtCompileTime,
                           OtherDerived,
                           Derived>;
public:
    template <typename OtherDerived>
    MatProdResult<OtherDerived>
    operator*(const EigenBase<OtherDerived>& e) const {
        return MatProdResult<OtherDerived>(
            derivedInner() * e.derivedInner()
        );
    }
    template <typename OtherDerived>
    MatProdResult<OtherDerived>
    operator*(const Eigen::EigenBase<OtherDerived>& e) const {
        return MatProdResult<OtherDerived>(
            derivedInner() * e
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
    template <int Mode, int Options>
    Transform<Scalar, Dim, Mode>
    operator*(const Eigen::Transform<Scalar, Dim, Mode, Options>& t) const {
        return Transform<Scalar, Dim, Mode>(derivedInner() * t);
    }

    // Compose with a translation
    Transform<Scalar, Dim, Isometry>
    operator*(const Translation<Scalar, Dim>& t) const {
        return Transform<Scalar, Dim, Isometry>(derivedInner() * t.getInner());
    }
    Transform<Scalar, Dim, Isometry>
    operator*(const Eigen::Translation<Scalar, Dim>& t) const {
        return Transform<Scalar, Dim, Isometry>(derivedInner() * t);
    }

    // Compose with a uniform scaling
    RotationMatrixType operator*(const UniformScaling<Scalar>& s) const {
        return RotationMatrixType(
            derivedInner() * s.getInner()
        );
    }

    // === Eagen-specific interface ===

    // CRTP daughter class access
    Derived& derived() {
        return *static_cast<Derived*>(this);
    }
    const Derived& derived() const {
        return *static_cast<const Derived*>(this);
    }

    // Access the inner Eigen object held by the CRTP daughter class
    Inner& derivedInner() {
        return derived().getInner();
    }
    const Inner& derivedInner() const {
        return derived().getInner();
    }
    Inner&& moveDerivedInner() {
        return derived().moveInner();
    }
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
