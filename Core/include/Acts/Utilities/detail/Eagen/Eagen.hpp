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
#include "AngleAxis.hpp"
#include "Block.hpp"
#include "CommaInitializer.hpp"
#include "EigenBase.hpp"
#include "EigenDense.hpp"
#include "EigenPrologue.hpp"
#include "ForwardDeclarations.hpp"
#include "Hyperplane.hpp"
#include "JacobiSVD.hpp"
#include "LDLT.hpp"
#include "LLT.hpp"
#include "Map.hpp"
#include "Matrix.hpp"
#include "MatrixBase.hpp"
#include "PlainMatrixBase.hpp"
#include "Quaternion.hpp"
#include "QuaternionBase.hpp"
#include "Rotation2D.hpp"
#include "RotationBase.hpp"
#include "SolverBase.hpp"
#include "Transform.hpp"
#include "Translation.hpp"
#include "TypeTraits.hpp"
#include "VectorBlock.hpp"

// These declarations must occur after everything else has been declared
#include "EigenEpilogue.hpp"
