// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EigenDense.hpp"
#include "EigenPrologue.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Wrapper of Eigen::Transform
template <typename _Scalar, int _Dim, int _Mode, int _Options>
class Transform {
public:
    // Expose template parameters
    using Scalar = _Scalar;
    static constexpr int Dim = _Dim;
    static constexpr int Mode = _Mode;
    static constexpr int Options = _Options;

    // Inner Eigen type
    using Inner = Eigen::Transform<Scalar, Dim, Mode, Options>;

    // Eigen-style typedefs and consts
    using Index = Eigen::Index;
    using RealScalar = typename Inner::RealScalar;
    using ConstLinearPart = Matrix<Scalar, Dim, Dim>;

    // TODO: Support affine()
    //       This requires quite a bit of work, and isn't used by Acts yet.

    // Cast to a different scalar type
    template <typename NewScalarType>
    Transform<NewScalarType, Dim, Mode, Options> cast() const {
        return Transform<NewScalarType, Dim, Mode, Options>(
            m_inner.template cast<NewScalarType>()
        );
    }

    // TODO: Support computeRotationScaling() and computeScalingRotation()
    //       This requires quite a bit of work, and isn't used by Acts yet.

    // Access inner data
    Scalar* data() {
        return m_inner.data();
    }
    const Scalar* data() const {
        return m_inner.data();
    }

    // TODO: Support fromPositionOrientationScale()
    //       This requires quite a bit of work, and isn't used by Acts yet.

    // Invert the transform
    Transform inverse(TransformTraits traits = static_cast<TransformTraits>(Mode)) const {
        return Transform(m_inner.inverse(traits));
    }

    // Approximate equality
    bool isApprox(const Transform& other,
                  const RealScalar& prec = dummy_precision()) const {
        return m_inner.isApprox(other.m_inner, prec);
    }
    bool isApprox(const Inner& other,
                  const RealScalar& prec = dummy_precision()) const {
        return m_inner.isApprox(other, prec);
    }

    // Linear part access
    //
    // FIXME: No in place access to the linear part yet, and thus no write
    //        access, because that's hard to do in a wrapper design and Acts
    //        doesn't use this Eigen feature yet.
    //
    ConstLinearPart linear() const {
        return ConstLinearPart(m_inner.linear());
    }

    // Set the last row to [0 ... 0 1]
    void makeAffine() {
        m_inner.makeAffine();
    }

    // TODO: Support rest of the Transform API. matrix() basically requires some
    //       kind of Map trick for in-place access, and this time I can't evade
    //       it with a dangerous FIXME as it's heavily used by Acts...

private:
    Inner m_inner;

    static RealScalar dummy_precision() {
        return Eigen::NumTraits<Scalar>::dummy_precision();
    }
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
