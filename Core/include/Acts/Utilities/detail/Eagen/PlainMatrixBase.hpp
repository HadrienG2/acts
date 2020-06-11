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

namespace Acts {

namespace detail {

namespace Eagen {

// Spiritual equivalent of the superclasses of Eigen::Matrix
//
// Since we do not currently provide an Eigen::Array wrapper, this is slightly
// specialized towards the needs of Eigen::Matrix, and therefore less general
// than Eigen::PlainObjectBase (hence the different name). But it can be turned
// into a true Eigen::PlainObjectBase equivalent if needed.
//
template <typename Derived>
class PlainMatrixBase : public MatrixBase<Derived> {
private:
    // Eigen type wrapped by the CRTP daughter class
    using Inner = typename MatrixBase<Derived>::Inner;

public:
    // Derived class scalar type
    using Scalar = typename MatrixBase<Derived>::Scalar;

    // === Eigen::PlainObjectBase interface ===

    // Resizing
    template <typename... Args>
    void conservativeResize(Args... args) {
        derivedInner().conservativeResize(args...);
    }
    template <typename OtherDerived>
    void conservativeResizeLike(const DenseBase<OtherDerived>& other) {
        derivedInner().conservativeResizeLike(other.derivedInner());
    }
    template <typename OtherDerived>
    void conservativeResizeLike(const Eigen::DenseBase<OtherDerived>& other) {
        derivedInner().conservativeResizeLike(other);
    }
    template <typename... Args>
    void resize(Args... args) {
        derivedInner().resize(args...);
    }
    template <typename OtherDerived>
    void resizeLike(const EigenBase<OtherDerived>& other) {
        derivedInner().resizeLike(other.derivedInner());
    }
    template <typename OtherDerived>
    void resizeLike(const Eigen::EigenBase<OtherDerived>& other) {
        derivedInner().resizeLike(other);
    }

    // Data access
    Scalar* data() { return derivedInner().data(); }
    const Scalar* data() const { return derivedInner().data(); }

    // Lazy assignment (?)
    template <typename OtherDerived>
    Derived& lazyAssign(const DenseBase<OtherDerived>& other) {
        derivedInner().lazyAssign(other.derivedInner());
        return *this;
    }
    template <typename OtherDerived>
    Derived& lazyAssign(const Eigen::DenseBase<OtherDerived>& other) {
        derivedInner().lazyAssign(other);
        return *this;
    }

    // Set inner values from various scalar sources
    template <typename... Args>
    Derived& setConstant(Args&&... args) {
        derivedInner().setConstant(std::forward<Args>(args)...);
        return *this;
    }
    template <typename... Index>
    Derived& setZero(Index... indices) {
        derivedInner().setZero(indices...);
        return *this;
    }
    template <typename... Index>
    Derived& setOnes(Index... indices) {
        derivedInner().setOnes(indices...);
        return *this;
    }
    template <typename... Index>
    Derived& setRandom(Index... indices) {
        derivedInner().setRandom(indices...);
        return *this;
    }

    // Map foreign data
    //
    // FIXME: In addition to being a possible performance issue, copying like
    //        this is also wrong at the semantic level, because Eigen's maps can
    //        be written to...
    //
    template <typename... Args>
    static Derived Map(Args&&... args) {
        return Derived(Inner::Map(std::forward<Args>(args)...));
    }
    template <typename... Args>
    static Derived MapAligned(Args&&... args) {
        return Derived(Inner::MapAligned(std::forward<Args>(args)...));
    }

protected:
    // FIXME: I have zero idea why this is apparently needed...
    using MatrixBase<Derived>::derivedInner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
