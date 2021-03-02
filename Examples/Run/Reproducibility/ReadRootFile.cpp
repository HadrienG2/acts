// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <RTypesCore.h>
#include <TBranch.h>
#include <TFile.h>
#include <TKey.h>
#include <TList.h>
#include <TObject.h>
#include <TTree.h>
#include <TTreeReader.h>

#include "ErrorHandling.hpp"
#include "ReadRootFile.hpp"


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
      if (keyCycle > latestCycleData.first) {
        latestCycleData = KeyCycle{ keyCycle, key };
      }
    } else {
      latestKeys.insert(std::move(keyName), KeyCycle{ keyCycle, key });
    }
  }

  // Process the latest keys
  for (const auto [keyName, keyCycle] : std::move(latestKeys)) {
    // Check that latest keys do map to TTrees and access them
    TKey& key = *keyCycle->second;
    TObject& obj = key.ReadObj();
    ASSERT_EQ(obj.ClassName(), "TTree", "Unsupported TKey type");
    TTree& tree = dynamic_cast<TTree&>(obj);

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

    // TODO: Finish translating compareRootFiles.C functionality
    // TODO: Recursively fill in this->keys
  }
}