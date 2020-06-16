// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EigenPrologue.hpp"
#include "TypeTraits.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Spiritual equivalent of Eigen::EigenBase
template <typename Derived>
class EigenBase {
protected:
    // Eigen type wrapped by the CRTP daughter class
    using DerivedTraits = TypeTraits<Derived>;
    using Inner = typename DerivedTraits::Inner;

public:
    // === Eigen::EigenBase interface ===

    // Index convenience typedef
    using Index = Eigen::Index;

    // Matrix dimensions
    Index cols() const {
        return derivedInner().cols();
    }
    Index rows() const {
        return derivedInner().rows();
    }
    Index size() const {
        return derivedInner().size();
    }

    // CRTP daughter class access
    Derived& derived() {
        return *static_cast<Derived*>(this);
    }
    const Derived& derived() const {
        return *static_cast<const Derived*>(this);
    }

    // === Eagen-specific interface ===

    // Access the inner Eigen object held by the CRTP daughter class
    //
    // FIXME: I'd like this implementation detail to be protected, but it seems
    //        that protected functions are not visible across different
    //        instantiations of MatrixBase, which I need to access Matrix
    //        contents from the Block implementation.
    //
    Inner& derivedInner() {
        return derived().getInner();
    }
    const Inner& derivedInner() const {
        return derived().getInner();
    }
    Inner&& moveDerivedInner() {
        return derived().moveInner();
    }
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts