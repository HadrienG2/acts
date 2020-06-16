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
#include "Matrix.hpp"
#include "MatrixBase.hpp"
#include "RotationBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Wrapper of Eigen::Rotation2D
template <typename _Scalar>
class Rotation2D : public RotationBase<Rotation2D<_Scalar>, 2> {
private:
    // RotationBase superclass
    using Super = RotationBase<Rotation2D<_Scalar>, 2>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Scalar = _Scalar;

    // Underlying Eigen type
    using Inner = Eigen::Rotation2D<Scalar>;

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

    // === Eigen::Rotation2D API ===

    // Default constructor
    Rotation2D() = default;

    // Constructor from 2D rotation matrix
    template <typename Derived>
    explicit Rotation2D(const MatrixBase<Derived>& m)
        : m_inner(m.derivedInner())
    {}
    template <typename Derived>
    explicit Rotation2D(const Eigen::MatrixBase<Derived>& m)
        : m_inner(m)
    {}

    // Constructor from other 2D rotation
    template <typename OtherScalarType>
    Rotation2D(const Rotation2D<OtherScalarType>& other)
        : m_inner(other.m_inner)
    {}
    template <typename OtherScalarType>
    Rotation2D(const Eigen::Rotation2D<OtherScalarType>& other)
        : m_inner(other)
    {}

    // Constructor from scalar angle
    explicit Rotation2D(const Scalar& a)
        : m_inner(a)
    {}

    // Rotation angle accessor
    Scalar& angle() {
        return m_inner.angle();
    }
    Scalar angle() const {
        return m_inner.angle();
    }

    // Scalar cast
    template <typename NewScalarType>
    Rotation2D<NewScalarType> cast() const {
        return Rotation2D<NewScalarType>(
            m_inner.template cast<NewScalarType>()
        );
    }

    // Assignment from rotation matrix
    template <typename Derived>
    Rotation2D& fromRotationMatrix(const MatrixBase<Derived>& mat) {
        m_inner.fromRotationMatrix(mat.derivedInner());
        return *this;
    }
    template <typename Derived>
    Rotation2D& fromRotationMatrix(const Eigen::MatrixBase<Derived>& mat) {
        m_inner.fromRotationMatrix(mat);
        return *this;
    }
    template <typename Derived>
    Rotation2D& operator=(const MatrixBase<Derived>& mat) {
        m_inner = mat.derivedInner();
        return *this;
    }
    template <typename Derived>
    Rotation2D& operator=(const Eigen::MatrixBase<Derived>& mat) {
        m_inner = mat;
        return *this;
    }

    // Convert to rotation matrix
    using Matrix2 = Matrix<Scalar, 2, 2>;
    Matrix2 toRotationMatrix() const {
        return Matrix2(m_inner.toRotationMatrix());
    }

    // Invert rotation
    Rotation2D inverse() const {
        return Rotation2D(m_inner.inverse());
    }

    // Approximate equality
    using RealScalar = typename Inner::RealScalar;
    bool isApprox(const Rotation2D& other,
                  const RealScalar& prec = dummy_precision()) const {
        return m_inner.isApprox(other.m_inner, prec);
    }
    bool isApprox(const Rotation2D& other,
                  const RealScalar& prec = dummy_precision()) const {
        return m_inner.isApprox(other, prec);
    }

    // Inherit multiplications from base class
    using Super::operator*;

    // 2D rotation product
    Rotation2D operator*(const Rotation2D& other) const {
        return Rotation2D(m_inner * other.m_inner);
    }
    Rotation2D operator*(const Inner& other) const {
        return Rotation2D(m_inner * other.m_inner);
    }
    Rotation2D& operator*=(const Rotation2D& other) {
        m_inner *= other.m_inner;
        return *this;
    }
    Rotation2D operator*=(const Inner& other) {
        m_inner *= other;
        return *this;
    }

    // Rotate a 2D vector
    using Vector2 = Vector<Scalar, 2>;
private:
    using Vector2Inner = typename Inner::Vector2;
public:
    Vector2 operator*(const Vector2& vec) {
        return Vector2(m_inner * vec.derivedInner());
    }
    Vector2 operator*(const Vector2Inner& vec) {
        return Vector2(m_inner * vec);
    }

    // Rotation interpolation
    Rotation2D<Scalar> slerp(const Scalar& t, const Rotation2D& other) const {
        return Rotation2D(m_inner.slerp(t, other.m_inner));
    }
    Rotation2D<Scalar> slerp(const Scalar& t, const Inner& other) const {
        return Rotation2D(m_inner.slerp(t, other));
    }

    // Inner angle wraparound
    Scalar smallestAngle() const {
        return m_inner.smallestAngle();
    }
    Scalar smallestPositiveAngle() const {
        return m_inner.smallestPositiveAngle();
    }

private:
    Inner m_inner;

private:
    static RealScalar dummy_precision() {
        return Eigen::NumTraits<Scalar>::dummy_precision();
    }
};

using Rotation2Dd = Rotation2D<double>;
using Rotation2Df = Rotation2D<float>;

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
