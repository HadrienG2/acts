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
template <typename _Derived>
class PlainMatrixBase : public MatrixBase<_Derived> {
    using Super = MatrixBase<_Derived>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Derived = _Derived;

    // Wrapped Eigen type
    using Inner = typename Super::Inner;

    // === Base class API ===

    // Re-export useful base class interface
protected:
    using DerivedTraits = typename Super::DerivedTraits;
public:
    using Index = typename Super::Index;
    using Scalar = typename Super::Scalar;
    using RealScalar = typename Super::RealScalar;
    using Super::derived;
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
        return derived();
    }
    template <typename OtherDerived>
    Derived& lazyAssign(const Eigen::DenseBase<OtherDerived>& other) {
        derivedInner().lazyAssign(other);
        return derived();
    }

    // Set inner values from various scalar sources
    template <typename... Args>
    Derived& setConstant(Args&&... args) {
        derivedInner().setConstant(std::forward<Args>(args)...);
        return derived();
    }
    template <typename... Index>
    Derived& setZero(Index... indices) {
        derivedInner().setZero(indices...);
        return derived();
    }
    template <typename... Index>
    Derived& setOnes(Index... indices) {
        derivedInner().setOnes(indices...);
        return derived();
    }
    template <typename... Index>
    Derived& setRandom(Index... indices) {
        derivedInner().setRandom(indices...);
        return derived();
    }

    // Map foreign data
private:
    template <typename Stride>
    using StridedMapType = Map<Derived, Unaligned, Stride>;
    template <typename Stride>
    using ConstStridedMapType = Map<const Derived, Unaligned, Stride>;
    template <typename Stride>
    using StridedAlignedMapType = Map<Derived, Aligned16, Stride>;
    template <typename Stride>
    using ConstStridedAlignedMapType = Map<const Derived, Aligned16, Stride>;
    using DefaultStride = Stride<DerivedTraits::OuterStride,
                                 DerivedTraits::InnerStride>;
    using MapType = StridedMapType<DefaultStride>;
    using ConstMapType = ConstStridedMapType<DefaultStride>;
    using AlignedMapType = StridedAlignedMapType<DefaultStride>;
    using ConstAlignedMapType = ConstStridedAlignedMapType<DefaultStride>;
public:
    static ConstMapType Map(const Scalar* data) {
        return ConstMapType(data);
    }
    static MapType Map(Scalar* data) {
        return MapType(data);
    }
    static ConstMapType Map(const Scalar* data, Index size) {
        return ConstMapType(data, size);
    }
    static MapType Map(Scalar* data, Index size) {
        return MapType(data, size);
    }
    static ConstMapType Map(const Scalar* data, Index rows, Index cols) {
        return ConstMapType(data, rows, cols);
    }
    static MapType Map(Scalar* data, Index rows, Index cols) {
        return MapType(data, rows, cols);
    }
    static ConstAlignedMapType MapAligned(const Scalar* data) {
        return ConstAlignedMapType(data);
    }
    static AlignedMapType MapAligned(Scalar* data) {
        return AlignedMapType(data);
    }
    static ConstAlignedMapType MapAligned(const Scalar* data, Index size) {
        return ConstAlignedMapType(data, size);
    }
    static AlignedMapType MapAligned(Scalar* data, Index size) {
        return AlignedMapType(data, size);
    }
    static ConstAlignedMapType MapAligned(const Scalar* data,
                                          Index rows,
                                          Index cols) {
        return ConstAlignedMapType(data, rows, cols);
    }
    static AlignedMapType MapAligned(Scalar* data, Index rows, Index cols) {
        return AlignedMapType(data, rows, cols);
    }
    template <int OuterStride, int InnerStride>
    static ConstStridedMapType<Stride<OuterStride, InnerStride>>
    Map(const Scalar* data, const Stride<OuterStride, InnerStride>& stride) {
        return ConstStridedMapType<Stride<OuterStride, InnerStride>>(
            data, stride);
    }
    template <int OuterStride, int InnerStride>
    static StridedMapType<Stride<OuterStride, InnerStride>>
    Map(Scalar* data, const Stride<OuterStride, InnerStride>& stride) {
        return StridedMapType<Stride<OuterStride, InnerStride>>(data, stride);
    }
    template <int OuterStride, int InnerStride>
    static ConstStridedMapType<Stride<OuterStride, InnerStride>>
    Map(const Scalar* data,
        Index size,
        const Stride<OuterStride, InnerStride>& stride) {
        return ConstStridedMapType<Stride<OuterStride, InnerStride>>(
            data, size, stride);
    }
    template <int OuterStride, int InnerStride>
    static StridedMapType<Stride<OuterStride, InnerStride>>
    Map(Scalar* data,
        Index size,
        const Stride<OuterStride, InnerStride>& stride)
    {
        return StridedMapType<Stride<OuterStride, InnerStride>>(
            data, size, stride);
    }
    template <int OuterStride, int InnerStride>
    static ConstStridedMapType<Stride<OuterStride, InnerStride>>
    Map(const Scalar* data,
        Index rows,
        Index cols,
        const Stride<OuterStride, InnerStride>& stride) {
        return ConstStridedMapType<Stride<OuterStride, InnerStride>>(
            data, rows, cols, stride);
    }
    template <int OuterStride, int InnerStride>
    static StridedMapType<Stride<OuterStride, InnerStride>>
    Map(Scalar* data,
        Index rows,
        Index cols,
        const Stride<OuterStride, InnerStride>& stride)
    {
        return StridedMapType<Stride<OuterStride, InnerStride>>(
            data, rows, cols, stride);
    }
    template <int OuterStride, int InnerStride>
    static ConstStridedAlignedMapType<Stride<OuterStride, InnerStride>>
    MapAligned(const Scalar* data,
               const Stride<OuterStride, InnerStride>& stride) {
        return ConstStridedAlignedMapType<Stride<OuterStride, InnerStride>>(
            data, stride);
    }
    template <int OuterStride, int InnerStride>
    static StridedAlignedMapType<Stride<OuterStride, InnerStride>>
    MapAligned(Scalar* data, const Stride<OuterStride, InnerStride>& stride) {
        return StridedAlignedMapType<Stride<OuterStride, InnerStride>>(
            data, stride);
    }
    template <int OuterStride, int InnerStride>
    static ConstStridedAlignedMapType<Stride<OuterStride, InnerStride>>
    MapAligned(const Scalar* data,
               Index size,
               const Stride<OuterStride, InnerStride>& stride) {
        return ConstStridedAlignedMapType<Stride<OuterStride, InnerStride>>(
            data, size, stride);
    }
    template <int OuterStride, int InnerStride>
    static StridedAlignedMapType<Stride<OuterStride, InnerStride>>
    MapAligned(Scalar* data,
               Index size,
               const Stride<OuterStride, InnerStride>& stride) {
        return StridedAlignedMapType<Stride<OuterStride, InnerStride>>(
            data, size, stride);
    }
    template <int OuterStride, int InnerStride>
    static ConstStridedAlignedMapType<Stride<OuterStride, InnerStride>>
    MapAligned(const Scalar* data,
               Index rows,
               Index cols,
               const Stride<OuterStride, InnerStride>& stride) {
        return ConstStridedAlignedMapType<Stride<OuterStride, InnerStride>>(
            data, rows, cols, stride);
    }
    template <int OuterStride, int InnerStride>
    static StridedAlignedMapType<Stride<OuterStride, InnerStride>>
    MapAligned(Scalar* data,
               Index rows,
               Index cols,
               const Stride<OuterStride, InnerStride>& stride) {
        return StridedAlignedMapType<Stride<OuterStride, InnerStride>>(
            data, rows, cols, stride);
    }
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
