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
#include "Matrix.hpp"
#include "MatrixBase.hpp"
#include "RotationBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Spiritual cousin of Eigen::QuaternionBase
template <typename _Derived>
class QuaternionBase : public RotationBase<_Derived, 3> {
    using Super = RotationBase<_Derived, 3>;

    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Derived = _Derived;

    // Wrapped Eigen type
    using Inner = typename Super::Inner;

    // Expose parameters of the child Quaternion type
    using Scalar = typename Super::Scalar;
protected:
    using DerivedTraits = typename Super::DerivedTraits;
public:
    static constexpr int Options = DerivedTraits::Options;

    // === Base class API ===

    // Re-export useful base class interface
    using Super::derived;
    using Super::derivedInner;
    using Super::operator*;

    // === Eigen::QuaternionBase API ===

    // Transform a vector
    using Vector3 = Vector<Scalar, 3>;
    Vector3 _transformVector(const Vector3& v) const {
        return Vector3(derivedInner()._transformVector(v.getInner()));
    }

    // Angular distance between quaternions
    template <typename OtherDerived>
    Scalar angularDistance(const QuaternionBase<OtherDerived>& other) const {
        return derivedInner().angularDistance(other.derivedInner());
    }

    // Scalar cast
    template <typename NewScalarType>
    Quaternion<NewScalarType> cast() const {
        return Quaternion<NewScalarType>(
            derivedInner().template cast<NewScalarType>()
        );
    }

    // Coefficient access
    using Coefficients = Matrix<Scalar, 4, 1, Options>;
private:
    using CoefficientsInner = typename Coefficients::Inner;
public:
    const Coefficients& coeffs() const {
        const auto& resultInner = derivedInner().coeffs();
        static_assert(
            std::is_same_v<decltype(resultInner), const CoefficientsInner&>,
            "Unexpected return type from Eigen in-place accessor");
        return reinterpret_cast<const Coefficients&>(resultInner);
    }
    Coefficients& coeffs() {
        auto& resultInner = derivedInner().coeffs();
        static_assert(
            std::is_same_v<decltype(resultInner), CoefficientsInner&>,
            "Unexpected return type from Eigen in-place accessor");
        return reinterpret_cast<Coefficients&>(resultInner);
    }
    VectorBlock<const Coefficients, 3> vec() const {
        return coeffs().template head<3>();
    }
    VectorBlock<Coefficients, 3> vec() {
        return coeffs().template head<3>();
    }
    Scalar w() const {
        return derivedInner().w();
    }
    Scalar& w() {
        return derivedInner().w();
    }
    Scalar x() const {
        return derivedInner().x();
    }
    Scalar& x() {
        return derivedInner().x();
    }
    Scalar y() const {
        return derivedInner().y();
    }
    Scalar& y() {
        return derivedInner().y();
    }
    Scalar z() const {
        return derivedInner().z();
    }
    Scalar& z() {
        return derivedInner().z();
    }

    // Complex conjugate
    Quaternion<Scalar> conjugate() const {
        return Quaternion<Scalar>(derivedInner().conjugate());
    }

    // Dot product
    template <typename OtherDerived>
    Scalar dot(const QuaternionBase<OtherDerived>& other) const {
        return derivedInner().dot(other.derivedInner());
    }

    // Inversion
    Quaternion<Scalar> inverse() const {
        return Quaternion<Scalar>(derivedInner().inverse());
    }

    // Approximate equality
private:
    using ScalarTraits = NumTraits<Scalar>;
public:
    using RealScalar = typename ScalarTraits::Real;
    template <typename OtherDerived>
    bool isApprox(const QuaternionBase<OtherDerived>& other,
                  const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isApprox(other.derivedInner(), prec);
    }

    // Norm and normalization
    Scalar norm() const {
        return derivedInner().norm();
    }
    void normalize() {
        derivedInner().normalize();
    }
    Quaternion<Scalar> normalized() const {
        return Quaternion<Scalar>(derivedInner().normalized());
    }
    Scalar squaredNorm() const {
        return derivedInner().squaredNorm();
    }

    // Quaternion product
    template <typename OtherDerived>
    Quaternion<Scalar>
    operator*(const QuaternionBase<OtherDerived>& other) const {
        return Quaternion<Scalar>(derivedInner() * other.derivedInner());
    }

    // In-place quaternion product
    template <typename OtherDerived>
    Derived& operator*=(const QuaternionBase<OtherDerived>& q) {
        derivedInner() *= q.derivedInner();
        return derived();
    }

    // Assignment from angle-axis transform
    Derived& operator=(const AngleAxis<Scalar>& aa) {
        derivedInner() = aa.getInner();
        return derived();
    }

    // Assignment from vector or matrix
    template <typename MatrixDerived>
    Derived& operator=(const MatrixBase<MatrixDerived>& mat) {
        derivedInner() = mat.derivedInner();
        return derived();
    }

    // Set from two vectors
    template <typename Derived1, typename Derived2>
    Derived& setFromTwoVectors(const MatrixBase<Derived1>& a,
                               const MatrixBase<Derived2>& b) {
        derivedInner().setFromTwoVectors(a.derivedInner(), b.derivedInner());
        return derived();
    }

    // Set to the identity quaternion
    QuaternionBase& setIdentity() {
        derivedInner().setIdentity();
        return *this;
    }

    // Quaternion linear interpolation
    template <typename OtherDerived>
    Quaternion<Scalar>
    slerp(const Scalar& t,
          const QuaternionBase<OtherDerived>& other) const {
        return Quaternion<Scalar>(
            derivedInner().slerp(t, other.derivedInner())
        );
    }

    // Convert to a rotation matrix
    using Matrix3 = Matrix<Scalar, 3, 3>;
    Matrix3 toRotationMatrix() const {
        return Matrix3(derivedInner().toRotationMatrix());
    }

    // Construct the identity quaternion
    static Quaternion<Scalar> Identity() {
        return Quaternion<Scalar>(Inner::Identity());
    }

private:
    static RealScalar dummy_precision() {
        return ScalarTraits::dummy_precision();
    }
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
