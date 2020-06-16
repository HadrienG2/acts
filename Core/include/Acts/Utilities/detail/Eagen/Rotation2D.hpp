// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EigenDense.hpp"
#include "MatrixBase.hpp"
#include "RotationBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Wrapper of Eigen::Rotation2D
template <typename _Scalar>
class Rotation2D : public RotationBase<Rotation2D<_Scalar>, 2> {
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

    // TODO: Remainder of the API

private:
    Inner m_inner;
};

using Rotation2Dd = Rotation2D<double>;
using Rotation2Df = Rotation2D<float>;

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
