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
#include "ForwardDeclarations.hpp"
#include "Matrix.hpp"
#include "MatrixBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Wrapper of Eigen::Hyperplane
template <typename _Scalar, int _AmbientDim, int _Options>
class Hyperplane {
public:
    // Eigen-style typedefs and constant propagation
    using Index = Eigen::Index;
    using Scalar = _Scalar;
    static constexpr int AmbientDim = _AmbientDim;
    static constexpr int Options = _Options;

    // Wrapped Eigen type
    using Inner = Eigen::Hyperplane<Scalar, AmbientDim, Options>;

    // More Eigen-style typedefs
    using RealScalar = typename Inner::RealScalar;
    using VectorType = Vector<Scalar, AmbientDim>;
private:
    using InnerVectorType = typename VectorType::Inner;
public:
    using Coefficients = Vector<Scalar,
                                (AmbientDim == Dynamic) ? Dynamic
                                                        : (AmbientDim + 1)>;

    // Default constructor
    Hyperplane() = default;

    // Constructor from another hyperplane
    template <typename OtherScalarType, int OtherOptions>
    Hyperplane(const Hyperplane<OtherScalarType, AmbientDim, OtherOptions>& other)
        : m_inner(other.m_inner)
    {}
    template <typename OtherScalarType, int OtherOptions>
    Hyperplane(const Eigen::Hyperplane<OtherScalarType, AmbientDim, OtherOptions>& other)
        : m_inner(other)
    {}

    // Constructor from a parametrized line
    // NOTE: No Eagen equivalent of this at this point in time
    Hyperplane(const Eigen::ParametrizedLine<Scalar, AmbientDim>& parametrized)
        : m_inner(parametrized)
    {}

    // Constructor from a vector and a scalar
    Hyperplane(const VectorType& n, const Scalar& d)
        : m_inner(n.getInner(), d)
    {}
    Hyperplane(const InnerVectorType& n, const Scalar& d)
        : m_inner(n, d)
    {}

    // Constructor from two vectors
    Hyperplane(const VectorType& n, const VectorType& e)
        : m_inner(n.getInner(), e.getInner())
    {}
    Hyperplane(const InnerVectorType& n, const VectorType& e)
        : m_inner(n, e.getInner())
    {}
    Hyperplane(const VectorType& n, const InnerVectorType& e)
        : m_inner(n.getInner(), e)
    {}
    Hyperplane(const InnerVectorType& n, const InnerVectorType& e)
        : m_inner(n, e)
    {}

    // Dynamic hyperplane constructor
    Hyperplane(Index dim) : m_inner(dim) {}

    // Distances from point to hyperplane
    Scalar absDistance(const VectorType& p) const {
        return m_inner.absDistance(p.getInner());
    }
    Scalar absDistance(const InnerVectorType& p) const {
        return m_inner.absDistance(p);
    }
    Scalar signedDistance(const VectorType& p) const {
        return m_inner.signedDistance(p.getInner());
    }
    Scalar signedDistance(const InnerVectorType& p) const {
        return m_inner.signedDistance(p);
    }

    // Cast to a different scalar type
    template <typename NewScalarType>
    Hyperplane<NewScalarType, AmbientDim, Options> cast() const {
        return Hyperplane<NewScalarType, AmbientDim, Options>(
            m_inner.template cast<NewScalarType>()
        );
    }

    // TODO: Support coeffs() accessor
    //       This requires a bit of work, and isn't used by Acts yet

    // Dimension of this hyperplane
    Index dim() const {
        return m_inner.dim();
    }

    // Intersection of two hyperplanes
    VectorType intersection(const Hyperplane& other) const {
        return VectorType(m_inner.intersection(other.m_inner));
    }
    VectorType intersection(const Inner& other) const {
        return VectorType(m_inner.intersection(other));
    }

    // Approximate hyperplane equality
    template <int OtherOptions>
    bool isApprox(const Hyperplane<Scalar, AmbientDim, OtherOptions>& other,
                  const RealScalar& prec = dummy_precision()) const {
        return m_inner.isApprox(other.m_inner, prec);
    }
    template <int OtherOptions>
    bool isApprox(const Eigen::Hyperplane<Scalar, AmbientDim, OtherOptions>& other,
                  const RealScalar& prec = dummy_precision()) const {
        return m_inner.isApprox(other, prec);
    }

    // TODO: Support normal() accessor
    //       This requires a bit of work, and isn't used by Acts yet

    // Normalize *this
    void normalize() {
        m_inner.normalize();
    }

    // Query distance to origin
    Scalar& offset() {
        return m_inner.offset();
    }
    const Scalar& offset() const {
        return m_inner.offset();
    }

    // Project a point on the hyperplane
    VectorType projection(const VectorType& p) const {
        return VectorType(m_inner.projection(p.getInner()));
    }
    VectorType projection(const InnerVectorType& p) const {
        return VectorType(m_inner.projection(p));
    }

    // Apply a transform to the hyperplane
    template <typename XprType>
    Hyperplane& transform(const MatrixBase<XprType>& mat,
                          TransformTraits traits = TransformTraits::Affine) {
        m_inner.transform(mat.derivedInner(), traits);
        return *this;
    }
    template <typename XprType>
    Hyperplane& transform(const Eigen::MatrixBase<XprType>& mat,
                          TransformTraits traits = TransformTraits::Affine) {
        m_inner.transform(mat, traits);
        return *this;
    }
    // NOTE: Eigen::Transform doesn't have an Eagen equivalent yet
    template <int TrOptions>
    Hyperplane& transform(const Eigen::Transform<Scalar,
                                                 AmbientDim,
                                                 TransformTraits::Affine,
                                                 TrOptions>& t,
                          TransformTraits traits = TransformTraits::Affine) {
        m_inner.transform(t, traits);
        return *this;
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
