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
#include "MatrixBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Eigen::Diagonal, but with eagerly evaluated operations
template <typename _MatrixType, int _DiagIndex>
class Diagonal : public MatrixBase<Diagonal<_MatrixType, _DiagIndex>> {
    using Super = MatrixBase<Diagonal>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using MatrixType = _MatrixType;
    static constexpr int DiagIndex = _DiagIndex;

    // Wrapped Eigen type
private:
    using MatrixTypeInner = typename MatrixType::Inner;
public:
    using Inner = Eigen::Diagonal<MatrixTypeInner, DiagIndex>;

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
    using Index = typename Super::Index;
    using Super::operator=;

    // === Eigen::Diagonal interface ===

    // Constructor from a matrix + dynamic diagonal index
    explicit Diagonal(MatrixType& matrix, Index a_index = DiagIndex)
        : m_inner(matrix.getInner())
    {}

private:
    Inner m_inner;
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
