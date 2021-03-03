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
#include <utility>
#include <vector>

#include <RTypesCore.h>
#include <TBranch.h>
#include <TDictionary.h>
#include <TKey.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>


// --- OUTPUT DATA ---

/// Useful type-agnostic properties of a ROOT TBranch
struct BranchProperties {
  std::string className;
  EDataType dataType;
  static constexpr Int_t numLeaves = 1;  // >1 leaves is not currently supported

  /// Extract a ROOT TBranch's properties
  BranchProperties(TBranch& branch);

  BranchProperties(BranchProperties&&) = default;
  BranchProperties& operator=(BranchProperties&&) = default;
};

/// Type-erased interface to TypedBranchData
struct BranchData : BranchProperties {
  /// Forward branch properties down the inheritance stack
  BranchData(BranchProperties&& properties)
    : BranchProperties{ std::move(properties) }
  {}

  virtual ~BranchData() = default;
  // TODO: More abstract interface to higher-level functionality
};

/// Concrete data from a ROOT TTree's branch
template <typename T>
struct TypedBranchData : BranchData {
  std::vector<T> data;

  /// Prepare to record a number of entries from a certain branch
  TypedBranchData(BranchProperties&& properties, size_t numEntries)
    : BranchData{ std::move(properties) }
  {
    data.reserve(numEntries);
  }

  // TODO: Implementation of the abstract BranchData API as it comes in
};

/// Data from a ROOT TTree
struct TreeData {
  size_t numEntries = 0;
  std::unordered_map<std::string, std::unique_ptr<BranchData>> branchData;

  /// Needed because constructors are evil
  TreeData() = default;

  /// Load data from a ROOT TTree
  TreeData(TTree& tree);
};

/// This code currently only supports TTree objects
using ObjectData = TreeData;

/// Data from the latest cycle of a set of identically named ROOT TKeys
///
/// ROOT files have an object versioning feature whose ergonomics are so bad
/// that it is almost only used by accident. Following the example of Cling, we
/// will therefore only consider the latest version of each named object...
///
struct KeyData {
  Int_t version;
  std::string className;
  std::string title;
  ObjectData data;

  /// Load data from a ROOT TKey
  KeyData(TKey& key);
}

/// Data from a ROOT file
struct FileData {
  Int_t version;
  std::unordered_map<std::string, KeyData> keys;

  /// Load data from a ROOT file
  FileData(const std::string& fileName);
};


// --- DATA READOUT PLUMBING ---

/// Type-erased interface to TypedBranchDataReader
class BranchDataReader {
public:
  virtual ~BranchDataReader() = default;

  /// Prepare to load data from a TTree branch
  static std::unique_ptr<BranchDataReader> setup(TTreeReader& treeReader,
                                                 size_t numEntries,
                                                 TBranch& branch);

  /// Record current branch data
  ///
  /// This method must be called for each recorded branch, after every call to
  /// the underlying TTreeReader's Next() method.
  ///
  virtual void collectValue() = 0;

  /// Dispose of this data readout harness and get the recorded data
  virtual std::unique_ptr<BranchData> finish() = 0;
};

/// Mechanism to read TypedBranchData using a TTreeReader
template <typename T>
class TypedBranchDataReader : public BranchDataReader {
public:
  TypedBranchDataReader(TTreeReader& reader,
                        size_t numEntries,
                        BranchProperties&& branchProperties,
                        const char* branchName)
    : m_data{ std::make_unique<TypedBranchData<T>>(std::move(branchProperties),
                                                   numEntries) }
    , m_reader{ reader, branchName }
  {}

  void collectValue() final override {
    m_data.data.push_back(*m_reader);
  }

  std::unique_ptr<BranchData> finish() final override {
    return std::move(m_data);
  }

private:
  std::unique_ptr<TypedBranchData<T>> m_data;
  TTreeReaderValue<T> m_reader;
}
