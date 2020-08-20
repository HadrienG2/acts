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
#include "ForwardDeclarations.hpp"
#include "Matrix.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Wrapper of Eigen::ParametrizedLine
template <typename _Scalar, int _AmbientDim, int _Options>
class ParametrizedLine {
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
    using Inner = Eigen::ParametrizedLine<Scalar, AmbientDim, Options>;

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

    // === Eigen::ParametrizedLine API ===

    // Good old Index typedef
    using Index = Eigen::Index;

    // More Eigen-style typedefs
    using VectorType = Vector<Scalar, AmbientDim>;

    // Default constructor
    ParametrizedLine() = default;

    // Constructor from inner Eigen type
    ParametrizedLine(const Inner& inner)
        : m_inner(inner)
    {}

    // Constructor from another parametrized line
    template <typename OtherScalarType, int OtherOptions>
    ParametrizedLine(const ParametrizedLine<OtherScalarType,
                                            AmbientDim,
                                            OtherOptions>& other)
        : m_inner(other.m_inner)
    {}

    // Constructor from an hyperplane
    template <int OtherOptions>
    explicit ParametrizedLine(const Hyperplane<Scalar,
                                               AmbientDim,
                                               OtherOptions>& hyperplane)
        : m_inner(hyperplane.getInner())
    {}

    // Constructor from two vectors
    ParametrizedLine(const VectorType& origin, const VectorType& direction)
        : m_inner(origin.getInner(), direction.getInner())
    {}

    // Dynamic parametrized line constructor
    explicit ParametrizedLine(Index dim) : m_inner(dim) {}

    // Cast to a different scalar type
    template <typename NewScalarType>
    ParametrizedLine<NewScalarType, AmbientDim, Options> cast() const {
        return ParametrizedLine<NewScalarType, AmbientDim, Options>(
            m_inner.template cast<NewScalarType>()
        );
    }

    // Dimension of this parametrized line
    Index dim() const {
        return m_inner.dim();
    }

    // Distance of a point to this line
    RealScalar distance(const VectorType& p) const {
        return m_inner.distance(p.getInner());
    }
    RealScalar squaredDistance(const VectorType& p) const {
        return m_inner.squaredDistance(p.getInner());
    }

    // NOTE: intersection() is not provided as it is deprecated & unused by Acts

    // Parameter value of the intersection between this line and an hyperplane
    template <int OtherOptions>
    Scalar intersectionParameter(
        const Hyperplane<Scalar, AmbientDim, OtherOptions>& hyperplane
    ) const {
        return m_inner.intersectionParameter(hyperplane.getInner());
    }

    // Point of intersection between this line and an hyperplane
    template <int OtherOptions>
    VectorType intersectionPoint(
        const Hyperplane<Scalar, AmbientDim, OtherOptions>& hyperplane
    ) const {
        return VectorType(m_inner.intersectionPoint(hyperplane.getInner()));
    }

    // Approximate parametrized line equality
    bool isApprox(const ParametrizedLine& other,
                  const RealScalar& prec = dummy_precision()) const {
        return m_inner.isApprox(other.m_inner, prec);
    }

    // Point at coordinate T along this line
    VectorType pointAt(const Scalar& t) const {
        return VectorType(m_inner.pointAt(t));
    }

    // Project a point on the parametrized line
    VectorType projection(const VectorType& p) const {
        return VectorType(m_inner.projection(p.getInner()));
    }

    // Construct a parametrized line passing through two points
    static ParametrizedLine Through(const VectorType& p0,
                                    const VectorType& p1) {
        return ParametrizedLine(Inner::Through(p0.getInner(), p1.getInner()));
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
