// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "AngleAxis.hpp"
#include "EigenDense.hpp"
#include "MatrixBase.hpp"
#include "QuaternionBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Wrapper of Eigen::Quaternion
template <typename _Scalar, int _Options>
class Quaternion : public QuaternionBase<Quaternion<_Scalar, _Options>> {
public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Scalar = _Scalar;
    static constexpr int Options = _Options;

    // Underlying Eigen type
    using Inner = Eigen::Quaternion<Scalar, Options>;

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

    // === Eigen::Quaternion API ===

    // Default constructor
    Quaternion() = default;

    // Constructor from an angle-axis rotation
    explicit Quaternion(const AngleAxis<Scalar>& aa) : m_inner(aa.getInner()) {}
    explicit Quaternion(const Eigen::AngleAxis<Scalar>& aa) : m_inner(aa) {}

    // Constructor from a vector
    template <typename Derived>
    explicit Quaternion(const MatrixBase<Derived>& other)
        : m_inner(other.derivedInner())
    {}
    template <typename Derived>
    explicit Quaternion(const Eigen::MatrixBase<Derived>& other)
        : m_inner(other)
    {}

    // Constructor from another quaternion
    template <typename OtherScalar, int OtherOptions>
    Quaternion(const Quaternion<OtherScalar, OtherOptions>& other)
        : m_inner(other.m_inner)
    {}
    template <typename OtherScalar, int OtherOptions>
    Quaternion(const Eigen::Quaternion<OtherScalar, OtherOptions>& other)
        : m_inner(other)
    {}

    // Constructor from QuaternionBase
    template <typename Derived>
    Quaternion(const QuaternionBase<Derived>& other)
        : m_inner(other.derivedInner())
    {}
    template <typename Derived>
    Quaternion(const Eigen::QuaternionBase<Derived>& other)
        : m_inner(other)
    {}

    // Constructor from four scalars
    Quaternion(const Scalar& w,
               const Scalar& x,
               const Scalar& y,
               const Scalar& z)
        : m_inner(w, x, y, z)
    {}

    // Constructor from a C-style array of scalars
    explicit Quaternion(const Scalar* data) : m_inner(data) {}

    // Generate a random normalized quaternion
    static Quaternion UnitRandom() {
        return Quaternion(Inner::UnitRandom());
    }

    // Generate a quaternion from two vectors
    template <typename Derived1, typename Derived2> 
    static Quaternion FromTwoVectors(const MatrixBase<Derived1>& a,
                                     const MatrixBase<Derived1>& b) {
        return Quaternion(
            Inner::FromTwoVectors(a.derivedInner(), b.derivedInner())
        );
    }
    template <typename Derived1, typename Derived2> 
    static Quaternion FromTwoVectors(const Eigen::MatrixBase<Derived1>& a,
                                     const MatrixBase<Derived1>& b) {
        return Quaternion(
            Inner::FromTwoVectors(a, b.derivedInner())
        );
    }
    template <typename Derived1, typename Derived2> 
    static Quaternion FromTwoVectors(const MatrixBase<Derived1>& a,
                                     const Eigen::MatrixBase<Derived1>& b) {
        return Quaternion(
            Inner::FromTwoVectors(a.derivedInner(), b)
        );
    }
    template <typename Derived1, typename Derived2> 
    static Quaternion FromTwoVectors(const Eigen::MatrixBase<Derived1>& a,
                                     const Eigen::MatrixBase<Derived1>& b) {
        return Quaternion(
            Inner::FromTwoVectors(a, b)
        );
    }

private:
    Inner m_inner;
};

using Quaterniond = Quaternion<double>;
using Quaternionf = Quaternion<float>;

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
