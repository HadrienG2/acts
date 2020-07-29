// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <utility>

#include "EigenDense.hpp"
#include "EigenPrologue.hpp"
#include "MatrixBase.hpp"
#include "RotationBase.hpp"
#include "Transform.hpp"
#include "Translation.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Wrapper of Eigen::UniformScaling
template<typename _Scalar>
class UniformScaling {
public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Scalar = _Scalar;

    // Wrapped Eigen type
    using Inner = Eigen::UniformScaling<Scalar>;

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

    // === Eigen::UniformScaling API ===

    // Eigen-style typedefs and consts
private:
    using ScalarTraits = NumTraits<Scalar>;
public:
    using RealScalar = typename ScalarTraits::Real;

    // Default constructor
    UniformScaling() = default;

    // Construct from inner Eigen type
    UniformScaling(const Inner& inner)
        : m_inner(inner)
    {}

    // Construct from a scalar
    explicit UniformScaling(const Scalar& s_) : m_inner(s_) {}

    // Construct from another uniform scaling
    template<typename OtherScalarType>
    explicit UniformScaling(const UniformScaling<OtherScalarType>& other)
        : m_inner(other.getInner())
    {}

    // Cast to a different scalar type
    template <typename NewScalarType>
    UniformScaling<NewScalarType> cast() const {
        return UniformScaling<NewScalarType>(
            m_inner.template cast<NewScalarType>()
        );
    }

    // Access scale factor
    Scalar factor() const {
        return m_inner.factor();
    }
    Scalar& factor() {
        return m_inner.factor();
    }

    // Concatenate two uniform scalings
    UniformScaling operator*(const UniformScaling& other) const {
        return UniformScaling(m_inner * other.getInner());
    }

    // Concatenate a uniform scaling and a translation
    template <int Dim>
    Transform<Scalar, Dim, Affine>
    operator*(const Translation<Scalar, Dim>& t) const {
        return Transform<Scalar, Dim, Affine>(
            m_inner * t.getInner()
        );
    }

    // Concatenate a uniform scaling and a linear transform
private:
    template <int Dim, int Mode>
    using ScalingTransformProdResult = Transform<
        Scalar,
        Dim,
        (Mode == (int)Isometry) ? Affine : Mode
    >;
public:
    template <int Dim, int Mode, int Options>
    ScalingTransformProdResult<Dim, Mode>
    operator*(const Transform<Scalar, Dim, Mode, Options>& t) const {
        return ScalingTransformProdResult<Dim, Mode>(
            m_inner * t.getInner()
        );
    }

    // Concatenate a uniform scaling and a rotation
    template <typename Derived,int Dim>
    Matrix<Scalar, Dim, Dim>
    operator*(const RotationBase<Derived, Dim>& r) const {
        return Matrix<Scalar, Dim, Dim>(
            m_inner * r.derivedInner()
        );
    }

    // Concatenate a linear transformation matrix and a uniform scaling
    template <typename Derived>
    friend Derived operator*(const MatrixBase<Derived>& matrix,
                             const UniformScaling& s) {
        return Derived(matrix * s.getInner());
    }

    // Inverse scaling
    UniformScaling inverse() const {
        return UniformScaling(
            m_inner.inverse()
        );
    }

    // Approximate equality
    bool isApprox(const UniformScaling& other,
                  const RealScalar& prec = dummy_precision()) const {
        return m_inner.isApprox(other.getInner(), prec);
    }

private:
    Inner m_inner;

    static RealScalar dummy_precision() {
        return ScalarTraits::dummy_precision();
    }
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
