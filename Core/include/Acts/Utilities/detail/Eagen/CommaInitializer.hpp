// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EigenDense.hpp"
#include "TypeTraits.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Comma initializer
template <typename Derived>
class CommaInitializer {
private:
    using DerivedTraits = TypeTraits<Derived>;

public:
    using Scalar = typename DerivedTraits::Scalar;

    // Constructor from (expression, scalar) pair
    CommaInitializer(Derived& derived, const Scalar& s)
        : m_derived(derived)
        , m_inner(derived.getInner(), s)
    {}

    // Constructor from (expression, expression) pair
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
    CommaInitializer& operator,(const Scalar& s) {
        m_inner, s;
        return *this;
    }

    // Expression insertion
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
    Eigen::CommaInitializer<typename DerivedTraits::Inner> m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
