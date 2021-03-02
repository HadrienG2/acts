// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <RTypesCore.h>
#include <TDictionary.h>


/// Abstract interface to BranchData + common subset of all BranchData types
struct BranchDataBase {
  std::string className;
  EDataType type;
  static constexpr Int_t nLeaves = 1;  // >1 leaves is not currently supported

  // TODO: Abstract interface to higher-level functionality
};

/// Concrete data that was extracted from a ROOT TTree's Branch
template <typename T>
struct BranchData : BranchDataBase {
  std::vector<T> data;

  // TODO: Implementation of the BranchDataBase interface
};

/// Data from a ROOT TTree
struct TreeData {
  size_t numEntries;
  std::unordered_map<std::string, std::unique_ptr<BranchDataBase>> branches;
};

/// This code currently only supports TTree data
using KeyData = TreeData;

/// Data from the latest cycle of a set of identically named ROOT TKeys
///
/// ROOT files have an object versioning feature whose ergonomics are so bad
/// that it is almost only used by accident. Following the example of Cling, we
/// will therefore only consider the latest version of each named object...
/// 
struct LatestKeyCycleData {
  Int_t version;
  std::string className;
  std::string title;
  KeyData data;
}

/// Data from a ROOT file
struct FileData {
  Int_t version;
  std::unordered_map<std::string, LatestKeyCycleData> keys;

  FileData(const std::string& fileName);
};