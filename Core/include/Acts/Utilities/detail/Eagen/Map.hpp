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
#include "PlainMatrixBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

template <typename Derived,
          int MapOptions = Unaligned,
          typename StrideType = Stride<TypeTraits<Derived>::OuterStride,
                                       TypeTraits<Derived>::InnerStride>>
class Map : public PlainMatrixBase<Map<Derived,
                                       MapOptions,
                                       StrideType>> {
private:
    using DerivedTraits = TypeTraits<Derived>;
    using DerivedInner = typename DerivedTraits::Inner;
    using Inner = Eigen::Map<DerivedInner, MapOptions, StrideType>;
    using PointerArgType = typename Inner::PointerArgType;
    using Index = Eigen::Index;

public:

    // Access the inner Eigen map
    Inner& getInner() {
        return m_inner;
    }
    const Inner& getInner() const {
        return m_inner;
    }

    // Eigen-like foreign data map constructors
    Map(PointerArgType dataPtr, const StrideType& stride = StrideType())
        : m_inner(dataPtr, stride)
    {}
    Map(PointerArgType dataPtr,
        Index rows,
        Index cols,
        const StrideType& stride = StrideType())
        : m_inner(dataPtr, rows, cols, stride)
    {}
    Map(PointerArgType dataPtr,
        Index size,
        const StrideType& stride = StrideType())
        : m_inner(dataPtr, size, stride)
    {}

private:
    Inner m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
