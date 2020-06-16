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
#include "MatrixBase.hpp"
#include "Matrix.hpp"
#include "QuaternionBase.hpp"
#include "RotationBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Wrapper of Eigen::AngleAxis
template <typename _Scalar>
class AngleAxis : public RotationBase<AngleAxis<_Scalar>, 3> {
private:
    using Super = RotationBase<AngleAxis<_Scalar>, 3>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Scalar = _Scalar;

    // Convenience typedef
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

    // Underlying Eigen type
    using Inner = Eigen::AngleAxis<Scalar>;

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

    // === Eigen::AngleAxis API ===

    // Default constructor
    AngleAxis() = default;

    // Constructor from another AngleAxis transform
    template <typename OtherScalar>
    AngleAxis(const AngleAxis<OtherScalar>& other)
        : m_inner(other.m_inner)
    {}
    template <typename OtherScalar>
    AngleAxis(const Eigen::AngleAxis<OtherScalar>& other)
        : m_inner(other)
    {}

    // Constructor from quaternion
    template <typename QuatDerived>
    explicit AngleAxis(const QuaternionBase<QuatDerived>& q)
        : m_inner(q.derivedInner())
    {}
    template <typename QuatDerived>
    explicit AngleAxis(const Eigen::QuaternionBase<QuatDerived>& q)
        : m_inner(q)
    {}

    // Constructor from angle and axis
    template <typename Derived>
    explicit AngleAxis(const Scalar& angle, const MatrixBase<Derived>& axis)
        : m_inner(angle, axis.derivedInner())
    {}
    template <typename Derived>
    explicit AngleAxis(const Scalar& angle,
                       const Eigen::MatrixBase<Derived>& axis)
        : m_inner(angle, axis)
    {}

    // Angle accessor
    Scalar& angle() {
        return m_inner.angle();
    }
    Scalar angle() const {
        return m_inner.angle();
    }

    // Axis accessor
private:
    using AxisType = Vector<Scalar, 3>;
    using AxisTypeMap = Map<AxisType>;
public:
    AxisType axis() const {
        return AxisType(m_inner.axis());
    }
    AxisTypeMap axis() {
        return AxisTypeMap(m_inner.axis().data());
    }

    // Scalar type casting
    template <typename NewScalarType>
    AngleAxis<NewScalarType> cast() const {
        return AngleAxis<NewScalarType>(m_inner.template cast<NewScalarType>());
    }

    // Convert to and from rotation matrix
    template <typename Derived>
    AngleAxis& fromRotationMatrix(const MatrixBase<Derived>& mat) {
        m_inner.fromRotationMatrix(mat.derivedInner());
    }
    template <typename Derived>
    AngleAxis& fromRotationMatrix(const Eigen::MatrixBase<Derived>& mat) {
        m_inner.fromRotationMatrix(mat);
    }
    Matrix<Scalar, 3, 3> toRotationMatrix() const {
        return Matrix<Scalar, 3, 3>(m_inner.toRotationMatrix());
    }

    // Compute the inverse transform
    AngleAxis inverse() const {
        AngleAxis(m_inner.inverse());
    }

    // Approximate equality
    bool isApprox(const AngleAxis& other,
                  const RealScalar& prec = dummy_precision()) const {
        return m_inner.isApprox(other.m_inner, prec);
    }
    bool isApprox(const Inner& other,
                  const RealScalar& prec = dummy_precision()) const {
        return m_inner.isApprox(other, prec);
    }

    // Angle-axis rotation composition
    using QuaternionType = Quaternion<Scalar>;
    QuaternionType operator*(const AngleAxis& other) const {
        return QuaternionType(m_inner * other.m_inner);
    }
    QuaternionType operator*(const Inner& other) const {
        return QuaternionType(m_inner * other);
    }
    using Super::operator*;

    // Assignment from matrix
    template <typename Derived>
    AngleAxis& operator=(const MatrixBase<Derived>& mat) {
        m_inner = mat.derivedInner();
        return *this;
    }
    template <typename Derived>
    AngleAxis& operator=(const Eigen::MatrixBase<Derived>& mat) {
        return *this;
    }

private:
    Inner m_inner;

    static RealScalar dummy_precision() {
        return Eigen::NumTraits<Scalar>::dummy_precision();
    }
};

using AngleAxisd = AngleAxis<double>;
using AngleAxisf = AngleAxis<float>;

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
