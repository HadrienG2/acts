// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EigenDense.hpp"
#include "ForwardDeclarations.hpp"
#include "Map.hpp"
#include "Matrix.hpp"
#include "MatrixBase.hpp"
#include "RotationBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Spiritual cousin of Eigen::QuaternionBase
template <typename Derived>
class QuaternionBase : public RotationBase<Derived, 3> {
private:
    // Bring some RotationBase typedefs into scope
    using Super = RotationBase<Derived, 3>;
    using Inner = typename Super::Inner;
    using DerivedTraits = typename Super::DerivedTraits;

public:
    // Derived class scalar type
    using Scalar = typename Super::Scalar;

    // Transform a vector
    using Vector3 = Vector<Scalar, 3>;
    Vector3 _transformVector(const Vector3& v) const {
        return Vector3(derivedInner()._transformVector(v.getInner()));
    }
private:
    using Vector3Inner = typename Vector3::Inner;
public:
    Vector3 _transformVector(const Vector3Inner& v) const {
        return Vector3(derivedInner()._transformVector(v));
    }

    // Angular distance between quaternions
    template <typename OtherDerived>
    Scalar angularDistance(const QuaternionBase<OtherDerived>& other) const {
        return derivedInner().angularDistance(other.derivedInner());
    }
    template <typename OtherDerived>
    Scalar
    angularDistance(const Eigen::QuaternionBase<OtherDerived>& other) const {
        return derivedInner().angularDistance(other);
    }

    // Scalar cast
    template <typename NewScalarType>
    Quaternion<NewScalarType> cast() const {
        return Quaternion<NewScalarType>(
            derivedInner().template cast<NewScalarType>()
        );
    }

    // Coefficient access
private:
    using Coefficients = Vector<Scalar, 4>;
    using CoefficientsMap = Map<Coefficients>;
public:
    Coefficients coeffs() const {
        return Coefficients(derivedInner().coeffs());
    }
    CoefficientsMap coeffs() {
        return CoefficientsMap(derivedInner().coeffs().data());
    }
private:
    using ImaginaryPart = Vector3;
    using ImaginaryPartMap = Map<Vector3>;
public:
    ImaginaryPart vec() const {
        return ImaginaryPart(derivedInner().vec());
    }
    ImaginaryPartMap vec() {
        return ImaginaryPartMap(derivedInner().vec().data());
    }
    const Scalar& w() const {
        return derivedInner().w();
    }
    Scalar& w() {
        return derivedInner().w();
    }
    const Scalar& x() const {
        return derivedInner().x();
    }
    Scalar& x() {
        return derivedInner().x();
    }
    const Scalar& y() const {
        return derivedInner().y();
    }
    Scalar& y() {
        return derivedInner().y();
    }
    const Scalar& z() const {
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
    template <typename OtherDerived>
    Scalar dot(const Eigen::QuaternionBase<OtherDerived>& other) const {
        return derivedInner().dot(other);
    }

    // Inversion
    Quaternion<Scalar> inverse() const {
        return Quaternion<Scalar>(derivedInner().inverse());
    }

    // Approximate equality
    using RealScalar = typename Inner::RealScalar;
    template <typename OtherDerived>
    bool isApprox(const QuaternionBase<OtherDerived>& other,
                  const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isApprox(other.derivedInner(), prec);
    }
    template <typename OtherDerived>
    bool isApprox(const Eigen::QuaternionBase<OtherDerived>& other,
                  const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isApprox(other, prec);
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
    template <typename OtherDerived>
    Quaternion<Scalar>
    operator*(const Eigen::QuaternionBase<OtherDerived>& other) const {
        return Quaternion<Scalar>(derivedInner() * other);
    }
    using Super::operator*;

    // In-place quaternion product
    template <typename OtherDerived>
    Derived& operator*=(const QuaternionBase<OtherDerived>& q) {
        derivedInner() *= q.derivedInner();
        return derived();
    }
    template <typename OtherDerived>
    Derived& operator*=(const Eigen::QuaternionBase<OtherDerived>& q) {
        derivedInner() *= q;
        return derived();
    }

    // Assignment from angle-axis transform
    Derived& operator=(const AngleAxis<Scalar>& aa) {
        derivedInner() = aa.getInner();
        return derived();
    }
    Derived& operator=(const Eigen::AngleAxis<Scalar>& aa) {
        derivedInner() = aa;
        return derived();
    }

    // Assignment from vector or matrix
    template <typename MatrixDerived>
    Derived& operator=(const MatrixBase<MatrixDerived>& mat) {
        derivedInner() = mat.derivedInner();
        return derived();
    }
    template <typename MatrixDerived>
    Derived& operator=(const Eigen::MatrixBase<MatrixDerived>& mat) {
        derivedInner() = mat;
        return derived();
    }

    // Set from two vectors
    template <typename Derived1, typename Derived2>
    Derived& setFromTwoVectors(const MatrixBase<Derived1>& a,
                               const MatrixBase<Derived2>& b) {
        derivedInner().setFromTwoVectors(a.derivedInner(), b.derivedInner());
        return derived();
    }
    template <typename Derived1, typename Derived2>
    Derived& setFromTwoVectors(const Eigen::MatrixBase<Derived1>& a,
                               const MatrixBase<Derived2>& b) {
        derivedInner().setFromTwoVectors(a, b.derivedInner());
        return derived();
    }
    template <typename Derived1, typename Derived2>
    Derived& setFromTwoVectors(const MatrixBase<Derived1>& a,
                               const Eigen::MatrixBase<Derived2>& b) {
        derivedInner().setFromTwoVectors(a.derivedInner(), b);
        return derived();
    }
    template <typename Derived1, typename Derived2>
    Derived& setFromTwoVectors(const Eigen::MatrixBase<Derived1>& a,
                               const Eigen::MatrixBase<Derived2>& b) {
        derivedInner().setFromTwoVectors(a, b);
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
    template <typename OtherDerived>
    Quaternion<Scalar>
    slerp(const Scalar& t,
          const Eigen::QuaternionBase<OtherDerived>& other) const {
        return Quaternion<Scalar>(
            derivedInner().slerp(t, other)
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

protected:
    // FIXME: I have zero idea why this is apparently needed...
    using Super::derived;
    using Super::derivedInner;

private:
    static RealScalar dummy_precision() {
        return Eigen::NumTraits<Scalar>::dummy_precision();
    }
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
