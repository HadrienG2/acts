// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <utility>

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
    using Super = QuaternionBase<Quaternion>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Scalar = _Scalar;
    static constexpr int Options = _Options;

    // Wrapped Eigen type
    using Inner = Eigen::Quaternion<Scalar, Options>;

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

    // === Eigen::Quaternion API ===

    // Default constructor
    Quaternion() = default;

    // Constructor from inner type
    Quaternion(const Inner& inner)
        : m_inner(inner)
    {}

    // Constructor from an angle-axis rotation
    explicit Quaternion(const AngleAxis<Scalar>& aa) : m_inner(aa.getInner()) {}

    // Constructor from a vector
    template <typename Derived>
    explicit Quaternion(const MatrixBase<Derived>& other)
        : m_inner(other.derivedInner())
    {}

    // Constructor from another quaternion
    template <typename OtherScalar, int OtherOptions>
    Quaternion(const Quaternion<OtherScalar, OtherOptions>& other)
        : m_inner(other.m_inner)
    {}

    // Constructor from QuaternionBase
    template <typename Derived>
    Quaternion(const QuaternionBase<Derived>& other)
        : m_inner(other.derivedInner())
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

private:
    Inner m_inner;
};

using Quaterniond = Quaternion<double>;
using Quaternionf = Quaternion<float>;

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
