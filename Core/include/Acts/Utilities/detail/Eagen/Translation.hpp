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
#include "EigenDense.hpp"
#include "ForwardDeclarations.hpp"
#include "Matrix.hpp"
#include "MatrixBase.hpp"
#include "RotationBase.hpp"
#include "Transform.hpp"
#include "UniformScaling.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Wrapper of Eigen::Translation
template <typename _Scalar, int _Dim>
class Translation {
public:
    // === Eagen wrapper API ===

    // Expose template parameters
    using Scalar = _Scalar;
    static constexpr int Dim = _Dim;

    // Inner Eigen type
    using Inner = Eigen::Translation<Scalar, Dim>;

    // Access the inner Eigen type
    Inner& getInner() {
        return m_inner;
    }
    const Inner& getInner() const {
        return m_inner;
    }

    // === Eigen::Translation API ===

    // Eigen-style typedefs
    using AffineTransformType = Transform<Scalar, Dim, Affine>;
    using IsometryTransformType = Transform<Scalar, Dim, Isometry>;
    using LinearMatrixType = Matrix<Scalar, Dim, Dim>;
    using VectorType = Vector<Scalar, Dim>;

    // Default constructor
    Translation() = default;

    // Constructor from another translation
    template <typename OtherScalarType>
    Translation(const Translation<OtherScalarType, Dim>& other)
        : m_inner(other.m_inner)
    {}
    template <typename OtherScalarType>
    Translation(const Eigen::Translation<OtherScalarType, Dim>& other)
        : m_inner(other)
    {}

    // /!\ UNDOCUMENTED /!\ constructors from scalars which Acts uses
    Translation(const Scalar& sx, const Scalar& sy)
        : m_inner(sx, sy)
    {}
    Translation(const Scalar& sx, const Scalar& sy, const Scalar& sz)
        : m_inner(sx, sy, sz)
    {}

    // Constructor from a translation vector
    explicit Translation(const VectorType& vector)
        : m_inner(vector.getInner())
    {}
private:
    using VectorTypeInner = typename VectorType::Inner;
public:
    explicit Translation(const VectorTypeInner& vector)
        : m_inner(vector)
    {}

    // Scalar cast
    template <typename NewScalarType>
    Translation<NewScalarType, Dim> cast() const {
        return Translation<NewScalarType, Dim>(
            m_inner.template cast<NewScalarType>()
        );
    }

    // Inverse translation
    Translation inverse() const {
        return Translation(m_inner.inverse());
    }

    // Approximate equality
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    bool isApprox(const Translation& other,
                  const RealScalar& prec = dummy_precision()) const {
        return m_inner.isApprox(other.m_inner, prec);
    }

    // Concatenate a translation and a linear transformation
    template <typename OtherDerived>
    AffineTransformType
    operator*(const EigenBase<OtherDerived>& linear) const {
        return AffineTransformType(m_inner * linear.derivedInner());
    }
    template <typename OtherDerived>
    AffineTransformType
    operator*(const Eigen::EigenBase<OtherDerived>& linear) const {
        return AffineTransformType(m_inner * linear);
    }

    // Left-hand-side multiplication by a matrix
    template<typename Derived> friend
    AffineTransformType operator*(const EigenBase<Derived>& l,
                                  const Translation& t)
    {
        return AffineTransformType(l.derivedInner() * t.getInner());
    }
    template<typename Derived> friend
    AffineTransformType operator*(const Eigen::EigenBase<Derived>& l,
                                  const Translation& t)
    {
        return AffineTransformType(l * t.getInner());
    }

    // Translate a vector
    template <typename Derived>
    std::enable_if_t<Derived::IsVectorAtCompileTime, VectorType>
    operator*(const MatrixBase<Derived>& vec) const {
        return m_inner * vec.derivedInner();
    }

    // Concatenate a translation and a rotation
    template <typename Derived>
    IsometryTransformType
    operator*(const RotationBase<Derived, Dim>& r) const {
        return IsometryTransformType(m_inner * r.derivedInner());
    }

    // Concatenate a translation and another transform
    template <int Mode, int Options>
    Transform<Scalar, Dim, Mode>
    operator*(const Transform<Scalar, Dim, Mode, Options>& t) const {
        return Transform<Scalar, Dim, Mode>(
            m_inner * t.getInner()
        );
    }

    // Concatenate two translations
    Translation operator*(const Translation& other) const {
        return Translation(m_inner * other.getInner());
    }

    // Concatenate a translation and a uniform scaling
    AffineTransformType operator*(const UniformScaling<Scalar>& other) const {
        return AffineTransformType(
            m_inner * other.getInner()
        );
    }

    // Coefficient access
    Scalar& x() {
        return m_inner.x();
    }
    Scalar x() const {
        return m_inner.x();
    }
    Scalar& y() {
        return m_inner.y();
    }
    Scalar y() const {
        return m_inner.y();
    }
    Scalar& z() {
        return m_inner.z();
    }
    Scalar z() const {
        return m_inner.z();
    }

private:
    Inner m_inner;

    static RealScalar dummy_precision() {
        return Eigen::NumTraits<Scalar>::dummy_precision();
    }
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
