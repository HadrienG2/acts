// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <TFile.h>
#include <TList.h>
#include <TObject.h>

#include "ErrorHandling.hpp"
#include "ReadRootFile.hpp"


BranchProperties(TBranch& branch)
  : className(branch.GetClassName())
{
  TClass* unused;
  branch.GetExpectedType(unused, this->dataType);
  ASSERT_EQ(branch.GetNleaves(), 1, "Multi-leaf branches are not supported");
}


TreeData::TreeData(TTree& tree)
  : numEntries(tree.GetEntries())
{
  // Read tree branches
  std::vector<TBranch*> branches;
  {
    const Int_t branchCount = tree.GetNbranches();
    branches.reserve(branchCount);
    TIter branchIter{tree.GetListOfBranches()};
    for (int i = 0; i < t1BranchCount; ++i) {
      branches.emplace_back(dynamic_cast<TBranch*>(branchIter()));
    }
  }

  // Prepare for tree readout
  TTreeReader treeReader(&tree);
  std::unordered_map<std::string, std::unique_ptr<BranchDataReader>> branchReaders;
  branchReaders.reserve(branches.size());
  for (const auto branchPtr : std::move(branches)) {
    branchReaders.push_back(
      branchPtr->GetName(),
      BranchDataReader::setup(treeReader, this->numEntries, *branchPtr)
    );
  }

  // Read event data
  for (size_t i = 0; i < this->numEntries; ++i) {
    treeReader.Next();
    for (auto& [_, branchReader] : branchReaders) {
      branchReader.collectValue();
    }
  }

  // Collect the event data
  this->branchData.reserve(branchReaders.size());
  for (auto&& [branchName, branchReader] : std::move(branchReaders)) {
    this->branchData.insert(branchName, branchReader.finish());
  }
}


KeyData::KeyData(TKey& key)
  : version(key.GetVersion())
  , className(key.GetClassName())
  , title(key.GetTitle())
{
  // Assert that the data is a TTree (only supported type at the moment)
  TObject& obj = key.ReadObj();
  ASSERT_EQ(obj.ClassName(), "TTree", "Unsupported TKey type");
  TTree& tree = dynamic_cast<TTree&>(obj);

  // Load data from the tree
  this->data = TreeData(tree);
}


FileData::FileData(const std::string& fileName) {
  // Open file
  TFile file(fileName.c_str());
  ASSERT(!file.IsZombie(), "Failed to open ROOT file \"" << fileName << '"');

  // Record file version
  this->version = file.GetVersion();

  // Read raw file keys
  std::vector<TKey*> fileKeys;
  {
    const Int_t keyCount = file.GetNkeys();
    fileKeys.reserve(keyCount);
    TIter keyIter{file.GetListOfKeys()};
    for (Int_t i = 0; i < keyCount; ++i) {
      fileKeys.push_back(dynamic_cast<TKey*>(keyIter()));
    }
  }

  // Group keys by name and select the latest key cycle
  using KeyCycle = std::pair<Short_t, TKey*>;
  std::unordered_map<std::string, KeyCycle> latestKeys;
  for (const auto key : std::move(fileKeys)) {
    // Extract key name and cycle number
    const std::string keyName = key->GetName();
    const Short_t keyCycle = key->GetCycle();

    // Create or update latest key cycle as appropriate
    auto latestCycleIter = latestKeys.find(keyName);
    if (latestCycleIter != latestKeys.end()) {
      KeyCycle& latestCycleData = latestCycleIter->second;
      ASSERT_NE(keyCycle, latestCycleData.first, "(cycle, name) pairs should be unique");
      if (keyCycle > latestCycleData.first) {
        latestCycleData = KeyCycle{ keyCycle, key };
      }
    } else {
      latestKeys.insert(std::move(keyName), KeyCycle{ keyCycle, key });
    }
  }

  // Load data from the latest cycle of each key
  for (const auto [keyName, keyCycle] : std::move(latestKeys)) {
    this->keys.emplace(keyName, *keyCycle->second);
  }
}


std::unique_ptr<BranchDataReader> setup(TTreeReader& treeReader,
                                        size_t numEntries,
                                        TBranch& branch) {
  BranchProperties branchProperties{ branch };
  const std::string& className = branchProperties.className;
  const char* const branchName = branch.GetName();

  // Unlike ROOT, we can't do runtime code generation, so we must define in
  // advance what kind of ROOT branch types we're going to support and
  // pre-instantiate TypedBranchDataReaders for all those types.
  #define INSTANTIATE(cppType) \
    return std::make_unique<TypedBranchDataReader<cppType>>( \
        treeReader, numEntries, std::move(branchProperties), branchName)

  #define TYPE_CASE(rootType, cppType) \
    case rootType: INSTANTIATE(cppType);

  switch (branchProperties.dataType) {
    TYPE_CASE(kChar_t, char)
    TYPE_CASE(kUChar_t, unsigned char)
    TYPE_CASE(kShort_t, short)
    TYPE_CASE(kUShort_t, unsigned short)
    TYPE_CASE(kInt_t, int)
    TYPE_CASE(kUInt_t, unsigned int)
    TYPE_CASE(kLong_t, long)
    TYPE_CASE(kULong_t, unsigned long)
    TYPE_CASE(kULong64_t, unsigned long long)
    TYPE_CASE(kFloat_t, float)
    TYPE_CASE(kDouble_t, double)
    TYPE_CASE(kBool_t, bool)
    case kOther_t:
      if (className.substr(0, 6) == "vector") {
        std::string elementType = className.substr(7, className.size() - 8);

        #define HANDLE_VECTOR(thisElementType) \
          if (elementType == #thisElementType) \
            INSTANTIATE(std::vector<thisElementType>) \
          else

        #define HANDLE_INTEGER_VECTOR(intElementType) \
          HANDLE_VECTOR(signed intElementType) \
          HANDLE_VECTOR(unsigned intElementType)

        HANDLE_INTEGER_VECTOR(char)
        HANDLE_INTEGER_VECTOR(short)
        HANDLE_INTEGER_VECTOR(int)
        HANDLE_INTEGER_VECTOR(long)
        HANDLE_INTEGER_VECTOR(long long)
        HANDLE_VECTOR(float)
        HANDLE_VECTOR(double)
        HANDLE_VECTOR(bool)
        THROW("Unsupported std::vector element type " << elementType);

        #undef HANDLE_INTEGER_VECTOR
        #undef HANDLE_VECTOR
      } else {
        THROW("Unsupported ROOT branch class " << className);
      }
    default:
      THROW("Unsupported ROOT branch type " << branchProperties.dataType);
  }

  #undef TYPE_CASE
  #undef INSTANTIATE
}
