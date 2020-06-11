// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <ostream>
#include <utility>

// Eagen is an eagerly evaluated Eigen wrapper, bypassing expression templates
#include "Block.hpp"
#include "EigenDense.hpp"
#include "EigenPrologue.hpp"
#include "ForwardDeclarations.hpp"
#include "TypeTraits.hpp"
#include "CommaInitializer.hpp"
#include "MatrixBase.hpp"
#include "PlainMatrixBase.hpp"
#include "Matrix.hpp"
#include "VectorBlock.hpp"

// These declarations must occur after everything else has been declared
#include "EigenEpilogue.hpp"
