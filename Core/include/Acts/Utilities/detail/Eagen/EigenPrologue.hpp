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
using AlignmentType = Eigen::AlignmentType;
constexpr AlignmentType Unaligned = Eigen::Unaligned;
constexpr AlignmentType Aligned8 = Eigen::Aligned8;
constexpr AlignmentType Aligned16 = Eigen::Aligned16;
constexpr AlignmentType Aligned32 = Eigen::Aligned32;
constexpr AlignmentType Aligned64 = Eigen::Aligned64;
constexpr AlignmentType Aligned128 = Eigen::Aligned128;
//
using ComputationInfo = Eigen::ComputationInfo;
constexpr ComputationInfo Success = Eigen::Success;
constexpr ComputationInfo NumericalIssue = Eigen::NumericalIssue;
constexpr ComputationInfo NoConvergence = Eigen::NoConvergence;
constexpr ComputationInfo InvalidInput = Eigen::InvalidInput;
//
using DecompositionOptions = Eigen::DecompositionOptions;
constexpr DecompositionOptions ComputeFullU = Eigen::ComputeFullU;
constexpr DecompositionOptions ComputeFullV = Eigen::ComputeFullV;
constexpr DecompositionOptions ComputeThinU = Eigen::ComputeThinU;
constexpr DecompositionOptions ComputeThinV = Eigen::ComputeThinV;
//
using Default_t = Eigen::Default_t;
constexpr Default_t Default = Eigen::Default;
//
constexpr int Dynamic = Eigen::Dynamic;
//
using NoChange_t = Eigen::NoChange_t;
constexpr NoChange_t NoChange = Eigen::NoChange;
//
using QRPreconditioners = Eigen::QRPreconditioners;
constexpr QRPreconditioners NoQRPreconditioner = Eigen::NoQRPreconditioner;
constexpr QRPreconditioners HouseholderQRPreconditioner =
    Eigen::HouseholderQRPreconditioner;
constexpr QRPreconditioners ColPivHouseholderQRPreconditioner =
    Eigen::ColPivHouseholderQRPreconditioner;
constexpr QRPreconditioners FullPivHouseholderQRPreconditioner =
    Eigen::FullPivHouseholderQRPreconditioner;
//
using StorageOptions = Eigen::StorageOptions;
constexpr StorageOptions AutoAlign = Eigen::AutoAlign;
constexpr StorageOptions ColMajor = Eigen::ColMajor;
constexpr StorageOptions RowMajor = Eigen::RowMajor;
//
template<int OuterStride, int InnerStride>
using Stride = Eigen::Stride<OuterStride, InnerStride>;
//
using TransformTraits = Eigen::TransformTraits;
constexpr TransformTraits Isometry = Eigen::Isometry;
constexpr TransformTraits Affine = Eigen::Affine;
constexpr TransformTraits AffineCompact = Eigen::AffineCompact;
constexpr TransformTraits Projective = Eigen::Projective;
//
using UpLoType = Eigen::UpLoType;
constexpr UpLoType Lower = Eigen::Lower;
constexpr UpLoType Upper = Eigen::Upper;
constexpr UpLoType UnitDiag = Eigen::UnitDiag;
constexpr UpLoType ZeroDiag = Eigen::ZeroDiag;
constexpr UpLoType UnitLower = Eigen::UnitLower;
constexpr UpLoType UnitUpper = Eigen::UnitUpper;
constexpr UpLoType StrictlyLower = Eigen::StrictlyLower;
constexpr UpLoType StrictlyUpper = Eigen::StrictlyUpper;
constexpr UpLoType SelfAdjoint = Eigen::SelfAdjoint;
constexpr UpLoType Symmetric = Eigen::Symmetric;

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
