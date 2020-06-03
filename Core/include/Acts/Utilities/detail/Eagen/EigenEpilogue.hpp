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
// Such definitions should preferably be put in the EigenPrologue.hpp header, so
// that the Eagen implementation can use them. EigenEpilogue.hpp should only be
// used for definitions that cannot be put in EigenPrologue.hpp because they
// clash with an identically named class-local typedef or constexpr.
//
using Index = Eigen::Index;

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
