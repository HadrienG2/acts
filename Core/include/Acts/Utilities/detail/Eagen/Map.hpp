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
#include "PlainMatrixBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

template <typename _Derived, int _MapOptions, typename _StrideType>
class Map : public PlainMatrixBase<Map<_Derived,
                                       _MapOptions,
                                       _StrideType>> {
    using Super = PlainMatrixBase<Map>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Derived = _Derived;
    static constexpr int MapOptions = _MapOptions;
    using StrideType = _StrideType;

    // Wrapped Eigen type
    using Inner = typename Super::Inner;

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

    // === Base class API ===

    // Re-export useful base class interface
    using Super::operator=;

    // === Eigen::Map API ===

    // Typedefs useful for constructors
    using PointerArgType = typename Inner::PointerArgType;
    using Index = typename Super::Index;

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
