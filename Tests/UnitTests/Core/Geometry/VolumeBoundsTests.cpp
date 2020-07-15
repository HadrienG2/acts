// This file is part of the Acts project.
//
// Copyright (C) 2019-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <boost/test/unit_test.hpp>

#include "Acts/Geometry/VolumeBounds.hpp"
#include "Acts/Utilities/Definitions.hpp"

namespace tt = boost::test_tools;

namespace Acts {

namespace Test {

BOOST_AUTO_TEST_SUITE(Volumes)

BOOST_AUTO_TEST_CASE(VolumeBoundsTest) {
  // Tests if the planes are correctly oriented
  // s_planeXY
  // s_planeYZ
  // s_planeZX

  Vector3D xaxis(1., 0., 0.);
  Vector3D yaxis(0., 1., 0.);
  Vector3D zaxis(0., 0., 1.);

  auto rotXY = s_planeXY.rotation();
  BOOST_CHECK(rotXY.extractCol(0).isApprox(xaxis));
  BOOST_CHECK(rotXY.extractCol(1).isApprox(yaxis));
  BOOST_CHECK(rotXY.extractCol(2).isApprox(zaxis));

  auto rotYZ = s_planeYZ.rotation();
  BOOST_CHECK(rotYZ.extractCol(0).isApprox(yaxis));
  BOOST_CHECK(rotYZ.extractCol(1).isApprox(zaxis));
  BOOST_CHECK(rotYZ.extractCol(2).isApprox(xaxis));

  auto rotZX = s_planeZX.rotation();
  BOOST_CHECK(rotZX.extractCol(0).isApprox(zaxis));
  BOOST_CHECK(rotZX.extractCol(1).isApprox(xaxis));
  BOOST_CHECK(rotZX.extractCol(2).isApprox(yaxis));
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace Test

}  // namespace Acts
