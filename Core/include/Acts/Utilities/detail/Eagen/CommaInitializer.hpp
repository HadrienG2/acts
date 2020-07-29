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
#include "TypeTraits.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Comma initializer
template <typename _Derived>
class CommaInitializer {
public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Derived = _Derived;

    // Wrapped Eigen type
private:
    using DerivedTraits = TypeTraits<Derived>;
    using DerivedInner = typename DerivedTraits::Inner;
public:
    using Inner = Eigen::CommaInitializer<DerivedInner>;

    // Access the inner Eigen object
    Inner& getInner() {
        return m_inner;
    }
    const Inner& getInner() const {
        return m_inner;
    }
    Inner&& moveInner() {
        return std::move(m_inner);
    }

    // === Eigen::CommaInitializer API ===

    // Inner scalar type
    using Scalar = typename DerivedTraits::Scalar;

    // Constructor from (expression, scalar) pair
    CommaInitializer(Derived& derived, const Scalar& s_)
        : m_derived(derived)
        , m_inner(derived.getInner(), s_)
    {}

    // Constructor from (expression, expression) pair
    //
    // NOTE: Remove Eigen::DenseBase overload if we ever wrap Eigen::Array.
    //
    template<typename OtherDerived>
    CommaInitializer(Derived& derived, const DenseBase<OtherDerived>& other)
        : CommaInitializer(derived, other.derivedInner())
    {}
    template<typename OtherDerived>
    CommaInitializer(Derived& derived,
                     const Eigen::DenseBase<OtherDerived>& other)
        : m_derived(derived)
        , m_inner(derived.getInner(), other)
    {}

    // Copy/Move constructor that transfers ownership
    //
    // NOTE: Added to be bugward compatible with Eigen, but the semantics of
    //       this API are just wrong in the post-C++11 era.
    //
    CommaInitializer(const CommaInitializer& o)
        : m_derived(o.m_derived)
        , m_inner(o.m_inner)
    {}

    // Scalar insertion
    CommaInitializer& operator,(const Scalar& s_) {
        m_inner, s_;
        return *this;
    }

    // Expression insertion
    //
    // NOTE: Remove Eigen::DenseBase overload if we ever wrap Eigen::Array.
    //
    template<typename OtherDerived>
    CommaInitializer& operator,(const DenseBase<OtherDerived>& other) {
        return *this, other.derivedInner();
    }
    template<typename OtherDerived>
    CommaInitializer& operator,(const Eigen::DenseBase<OtherDerived>& other) {
        m_inner, other;
        return *this;
    }

    // Get the built matrix
    Derived& finished() {
        m_inner.finished();
        return m_derived;
    }

private:
    Derived& m_derived;
    Inner m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
