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
#include "ForwardDeclarations.hpp"
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
    using Super = RotationBase<AngleAxis, 3>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Scalar = _Scalar;

    // Convenience typedefs
    using ScalarTraits = NumTraits<Scalar>;
    using RealScalar = typename ScalarTraits::Real;

    // Wrapped Eigen type
    using Inner = Eigen::AngleAxis<Scalar>;

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

    // === Eigen::AngleAxis API ===

    // Default constructor
    AngleAxis() = default;

    // Constructor from inner Eigen type
    AngleAxis(const Inner& inner)
        : m_inner(inner)
    {}

    // Constructor from another AngleAxis transform
    template <typename OtherScalar>
    AngleAxis(const AngleAxis<OtherScalar>& other)
        : m_inner(other.m_inner)
    {}

    // Constructor from quaternion
    template <typename QuatDerived>
    explicit AngleAxis(const QuaternionBase<QuatDerived>& q)
        : m_inner(q.derivedInner())
    {}

    // Constructor from angle and axis
    template <typename Derived>
    explicit AngleAxis(const Scalar& angle, const MatrixBase<Derived>& axis)
        : m_inner(angle, axis.derivedInner())
    {}

    // Angle accessor
    Scalar& angle() {
        return m_inner.angle();
    }
    Scalar angle() const {
        return m_inner.angle();
    }

    // Axis accessor
    using Vector3 = Vector<Scalar, 3>;
private:
    using Vector3Inner = typename Vector3::Inner;
public:
    const Vector3& axis() const {
        const auto& resultInner = m_inner.axis();
        static_assert(
            std::is_same_v<decltype(resultInner), const Vector3Inner&>,
            "Unexpected return type from Eigen in-place accessor");
        return reinterpret_cast<const Vector3&>(resultInner);
    }
    Vector3& axis() {
        auto& resultInner = m_inner.axis();
        static_assert(
            std::is_same_v<decltype(resultInner), Vector3Inner&>,
            "Unexpected return type from Eigen in-place accessor");
        return reinterpret_cast<Vector3&>(resultInner);
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

    // Angle-axis rotation composition
    using QuaternionType = Quaternion<Scalar>;
    QuaternionType operator*(const AngleAxis& other) const {
        return QuaternionType(m_inner * other.m_inner);
    }
    friend QuaternionType operator*(const QuaternionType& a,
                                    const AngleAxis& b) {
        return QuaternionType(a.derivedInner() * b.m_inner);
    }
    using Super::operator*;

    // Assignment from matrix
    template <typename Derived>
    AngleAxis& operator=(const MatrixBase<Derived>& mat) {
        m_inner = mat.derivedInner();
        return *this;
    }

private:
    Inner m_inner;

    static RealScalar dummy_precision() {
        return ScalarTraits::dummy_precision();
    }
};

using AngleAxisd = AngleAxis<double>;
using AngleAxisf = AngleAxis<float>;

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
