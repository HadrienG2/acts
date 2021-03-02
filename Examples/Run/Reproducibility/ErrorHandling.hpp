// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <exception>
#include <sstream>


#define THROW(msg) \
  do { \
    std::ostringstream os; \
    os << msg; \
    throw std::runtime_error(os.str()); \
  } while(false)


#define ASSERT(pred, msg) \
  if (!(pred)) THROW(msg << " (assertion " << #pred << " failed)")


#define COMPARISON_FAILURE(ref, val, msg) \
  THROW(msg << " (observed value " << #val << " (" << (val) << ") "\
        << " didn't match reference " << #ref << " (" << (ref) << "))")


#define ASSERT_EQ(ref, val, msg) \
  if ((val) != (ref)) COMPARISON_FAILURE(ref, val, msg)


#define ASSERT_NE(ref, val, msg) \
  if ((val) == (ref)) COMPARISON_FAILURE(ref, val, msg)