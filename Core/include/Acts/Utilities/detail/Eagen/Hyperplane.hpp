// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <utility>

#include "EigenDense.hpp"
#include "EigenPrologue.hpp"
#include "ForwardDeclarations.hpp"
#include "Matrix.hpp"
#include "MatrixBase.hpp"
#include "Transform.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Wrapper of Eigen::Hyperplane
template <typename _Scalar, int _AmbientDim, int _Options>
class Hyperplane {
public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Scalar = _Scalar;
    static constexpr int AmbientDim = _AmbientDim;
    static constexpr int Options = _Options;

    // Convenience typedefs
private:
    using ScalarTraits = NumTraits<Scalar>;
public:
    using RealScalar = typename ScalarTraits::Real;

    // Wrapped Eigen type
    using Inner = Eigen::Hyperplane<Scalar, AmbientDim, Options>;

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

    // === Eigen::Hyperplane API ===

    // Good old Index typedef
    using Index = Eigen::Index;

    // More Eigen-style typedefs
    using VectorType = Vector<Scalar, AmbientDim>;

    // Default constructor
    Hyperplane() = default;

    // Constructor from inner Eigen type
    Hyperplane(const Inner& inner)
        : m_inner(inner)
    {}

    // Constructor from another hyperplane
    template <typename OtherScalarType, int OtherOptions>
    Hyperplane(const Hyperplane<OtherScalarType,
                                AmbientDim,
                                OtherOptions>& other)
        : m_inner(other.m_inner)
    {}

    // Constructor from a parametrized line
    // NOTE: No Eagen equivalent of this at this point in time
    explicit Hyperplane(const Eigen::ParametrizedLine<Scalar,
                                                      AmbientDim>& parametrized)
        : m_inner(parametrized)
    {}

    // Constructor from a vector and a scalar
    Hyperplane(const VectorType& n, const Scalar& d)
        : m_inner(n.getInner(), d)
    {}

    // Constructor from two vectors
    Hyperplane(const VectorType& n, const VectorType& e)
        : m_inner(n.getInner(), e.getInner())
    {}

    // Dynamic hyperplane constructor
    explicit Hyperplane(Index dim) : m_inner(dim) {}

    // Distances from point to hyperplane
    Scalar absDistance(const VectorType& p) const {
        return m_inner.absDistance(p.getInner());
    }
    Scalar signedDistance(const VectorType& p) const {
        return m_inner.signedDistance(p.getInner());
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

    // Approximate hyperplane equality
    template <int OtherOptions>
    bool isApprox(const Hyperplane<Scalar, AmbientDim, OtherOptions>& other,
                  const RealScalar& prec = dummy_precision()) const {
        return m_inner.isApprox(other.m_inner, prec);
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

    // Apply a transform to the hyperplane
    template <typename XprType>
    Hyperplane& transform(const MatrixBase<XprType>& mat,
                          TransformTraits traits = TransformTraits::Affine) {
        m_inner.transform(mat.derivedInner(), traits);
        return *this;
    }
    template <int TrOptions>
    Hyperplane& transform(const Transform<Scalar,
                                          AmbientDim,
                                          TransformTraits::Affine,
                                          TrOptions>& t,
                          TransformTraits traits = TransformTraits::Affine) {
        m_inner.transform(t.getInner(), traits);
        return *this;
    }

    // Construct a hyperplane passing through two points
    static Hyperplane Through(const VectorType& p0, const VectorType& p1) {
        return Hyperplane(Inner::Through(p0.getInner(), p1.getInner()));
    }

    // Construct a hyperplane passing through three points
    static Hyperplane Through(const VectorType& p0,
                              const VectorType& p1,
                              const VectorType& p2) {
        return Hyperplane(
            Inner::Through(p0.getInner(), p1.getInner(), p2.getInner())
        );
    }

private:
    Inner m_inner;

    static RealScalar dummy_precision() {
        return ScalarTraits::dummy_precision();
    }
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
