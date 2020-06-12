// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EigenDense.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Propagate some Eigen types and constants
//
// If a type or constant cannot be propagated to the Eagen namespace right away
// because it would conflict with a class-local typedef or constexpr with the
// same name, put it in EigenEpilogue.hpp
//
constexpr int AutoAlign = Eigen::AutoAlign;
constexpr int ColMajor = Eigen::ColMajor;
constexpr int Dynamic = Eigen::Dynamic;
constexpr int RowMajor = Eigen::RowMajor;
//
using NoChange_t = Eigen::NoChange_t;
constexpr NoChange_t NoChange = Eigen::NoChange;
//
using TransformTraits = Eigen::TransformTraits;
constexpr TransformTraits Isometry = Eigen::Isometry;
constexpr TransformTraits Affine = Eigen::Affine;
constexpr TransformTraits AffineCompact = Eigen::AffineCompact;
constexpr TransformTraits Projective = Eigen::Projective;

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
