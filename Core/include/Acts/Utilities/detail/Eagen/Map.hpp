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
#include "PlainMatrixBase.hpp"
#include "TypeTraits.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

template <typename Derived, int MapOptions, typename StrideType>
class Map : public PlainMatrixBase<Map<Derived,
                                       MapOptions,
                                       StrideType>> {
private:
    using Super = PlainMatrixBase<Map>;
    using DerivedTraits = TypeTraits<Derived>;
    using DerivedInner = typename DerivedTraits::Inner;
    using SelfTraits = TypeTraits<Map>;
    using Inner = typename SelfTraits::Inner;
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
    Inner&& moveInner() {
        return std::move(m_inner);
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

    // Inherit useful base class facilities
    using Super::operator=;

private:
    Inner m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
