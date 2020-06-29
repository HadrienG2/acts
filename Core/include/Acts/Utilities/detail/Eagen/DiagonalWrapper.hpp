// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <utility>

#include "DiagonalBase.hpp"
#include "EigenDense.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

template <typename _DiagonalVectorType>
class DiagonalWrapper
    : public DiagonalBase<DiagonalWrapper<_DiagonalVectorType>>
{
    using Super = DiagonalBase<DiagonalWrapper>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using DiagonalVectorType = _DiagonalVectorType;

    // Wrapped Eigen type
    using Inner = typename Super::Inner;

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

    // === Base class API ===

    // Re-export useful base class interface
    using Index = typename Super::Index;
    using Super::diagonal;

    // === Eigen::DiagonalWrapper API ===

    // Constructor from expression of diagonal coefficients
    explicit DiagonalWrapper(DiagonalVectorType& a_diagonal)
        : m_inner(a_diagonal.getInner())
    {}

private:
    Inner m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
