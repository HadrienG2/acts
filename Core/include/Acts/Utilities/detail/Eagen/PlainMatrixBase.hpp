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
    // Bring some MatrixBase typedefs into scope
    using Super = MatrixBase<Derived>;
    using Inner = typename Super::Inner;
    using Index = typename Super::Index;
    using DerivedTraits = typename Super::DerivedTraits;

public:
    // Re-expose superclass interface
    using Scalar = typename Super::Scalar;
    using Super::derivedInner;
    using Super::operator=;

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
private:
    template <typename Stride>
    using StridedMapType = Map<Derived, Unaligned, Stride>;
    template <typename Stride>
    using StridedAlignedMapType = Map<Derived, Aligned16, Stride>;
    using DefaultStride = Stride<DerivedTraits::OuterStride,
                                 DerivedTraits::InnerStride>;
    using MapType = StridedMapType<DefaultStride>;
    using AlignedMapType = StridedAlignedMapType<DefaultStride>;
public:
    template <typename... Args>
    static Derived Map(const Scalar* data, Args&&... args) {
        return Derived(Inner::Map(data, std::forward<Args>(args)...));
    }
    template <typename... Args>
    static Derived MapAligned(const Scalar* data, Args&&... args) {
        return Derived(Inner::MapAligned(data, std::forward<Args>(args)...));
    }
    static MapType Map(Scalar* data) {
        return MapType(data);
    }
    static MapType Map(Scalar* data, Index size) {
        return MapType(data, size);
    }
    static MapType Map(Scalar* data, Index rows, Index cols) {
        return MapType(data, rows, cols);
    }
    static AlignedMapType MapAligned(Scalar* data) {
        return AlignedMapType(data);
    }
    static AlignedMapType MapAligned(Scalar* data, Index size) {
        return AlignedMapType(data, size);
    }
    static AlignedMapType MapAligned(Scalar* data, Index rows, Index cols) {
        return AlignedMapType(data, rows, cols);
    }
    template <int Outer, int Inner>
    static StridedMapType<Stride<Outer, Inner>>
    Map(Scalar* data, const Stride<Outer, Inner>& stride) {
        return StridedMapType<Stride<Outer, Inner>>(data, stride);
    }
    template <int Outer, int Inner>
    static StridedMapType<Stride<Outer, Inner>>
    Map(Scalar* data, Index size, const Stride<Outer, Inner>& stride) {
        return StridedMapType<Stride<Outer, Inner>>(data, size, stride);
    }
    template <int Outer, int Inner>
    static StridedMapType<Stride<Outer, Inner>>
    Map(Scalar* data,
        Index rows,
        Index cols,
        const Stride<Outer, Inner>& stride)
    {
        return StridedMapType<Stride<Outer, Inner>>(data, rows, cols, stride);
    }
    template <int Outer, int Inner>
    static StridedAlignedMapType<Stride<Outer, Inner>>
    MapAligned(Scalar* data, const Stride<Outer, Inner>& stride) {
        return StridedAlignedMapType<Stride<Outer, Inner>>(data, stride);
    }
    template <int Outer, int Inner>
    static StridedAlignedMapType<Stride<Outer, Inner>>
    MapAligned(Scalar* data, Index size, const Stride<Outer, Inner>& stride) {
        return StridedAlignedMapType<Stride<Outer, Inner>>(data, size, stride);
    }
    template <int Outer, int Inner>
    static StridedAlignedMapType<Stride<Outer, Inner>>
    MapAligned(Scalar* data,
               Index rows,
               Index cols,
               const Stride<Outer, Inner>& stride) {
        return StridedAlignedMapType<Stride<Outer, Inner>>(data,
                                                           rows,
                                                           cols,
                                                           stride);
    }
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
