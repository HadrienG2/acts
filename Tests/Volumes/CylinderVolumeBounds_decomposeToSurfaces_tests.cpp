// This file is part of the ACTS project.
//
// Copyright (C) 2016 ACTS project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define BOOST_TEST_MODULE Cylinder Volume Bounds Tests
#include <boost/test/included/unit_test.hpp>

#include <boost/test/data/test_case.hpp>
#include "ACTS/Surfaces/Surface.hpp"
#include "ACTS/Utilities/Definitions.hpp"
#include "ACTS/Volumes/CylinderVolumeBounds.hpp"

namespace bdata = boost::unit_test::data;
namespace tt    = boost::test_tools;

namespace Acts {

namespace Test {

  /// Unit test for testing the decomposeToSurfaces() function
  BOOST_DATA_TEST_CASE(CylinderVolumeBounds_decomposeToSurfaces,
                       bdata::random(-M_PI, M_PI) ^ bdata::random(-M_PI, M_PI)
                           ^ bdata::random(-M_PI, M_PI)
                           ^ bdata::random(-10., 10.)
                           ^ bdata::random(-10., 10.)
                           ^ bdata::random(-10., 10.)
                           ^ bdata::xrange(100),
                       alpha,
                       beta,
                       gamma,
                       posX,
                       posY,
                       posZ,
                       index)
  {
    // position of volume
    const Vector3D pos(posX, posY, posZ);
    // rotation around x axis
    AngleAxis3D rotX(alpha, Vector3D(1., 0., 0.));
    // rotation around y axis
    AngleAxis3D rotY(beta, Vector3D(0., 1., 0.));
    // rotation around z axis
    AngleAxis3D rotZ(gamma, Vector3D(0., 0., 1.));

    // create the cylinder bounds
    CylinderVolumeBounds cylBounds(1., 2., 3.);
    // create the transformation matrix
    auto transformPtr = std::make_shared<Transform3D>(pos);
    (*transformPtr) *= rotZ;
    (*transformPtr) *= rotY;
    (*transformPtr) *= rotX;
    // get the boundary surfaces
    const std::vector<const Acts::Surface*> boundarySurfaces
        = cylBounds.decomposeToSurfaces(transformPtr);
    // Test

    // check if difference is halfZ - sign and direction independent
    BOOST_TEST((pos - boundarySurfaces.at(0)->center()).mag()
                   == cylBounds.halflengthZ(),
               tt::tolerance(10e-12));
    BOOST_TEST((pos - boundarySurfaces.at(1)->center()).mag()
                   == cylBounds.halflengthZ(),
               tt::tolerance(10e-12));
    // transform to local
    double posDiscPosZ
        = (transformPtr->inverse() * boundarySurfaces.at(1)->center()).z();
    double centerPosZ = (transformPtr->inverse() * pos).z();
    double negDiscPosZ
        = (transformPtr->inverse() * boundarySurfaces.at(0)->center()).z();
    // check if center of disc boundaries lies in the middle in z
    BOOST_TEST((centerPosZ < posDiscPosZ));
    BOOST_TEST((centerPosZ > negDiscPosZ));
    // check positions of disc boundarysurfaces
    BOOST_TEST(negDiscPosZ + cylBounds.halflengthZ()
                   == (transformPtr->inverse() * pos).z(),
               tt::tolerance(10e-12));
    BOOST_TEST(posDiscPosZ - cylBounds.halflengthZ()
                   == (transformPtr->inverse() * pos).z(),
               tt::tolerance(10e-12));
    // orientation of disc surfaces
    // positive disc durface should point in positive direction in the frame of
    // the volume
    BOOST_TEST(transformPtr->rotation().col(2).dot(
                   boundarySurfaces.at(1)->normal(Acts::Vector2D(0., 0.)))
                   == 1.,
               tt::tolerance(10e-12));
    // negative disc durface should point in negative direction in the frame of
    // the volume
    BOOST_TEST(transformPtr->rotation().col(2).dot(
                   boundarySurfaces.at(0)->normal(Acts::Vector2D(0., 0.)))
                   == -1.,
               tt::tolerance(10e-12));
    // test in r
    BOOST_TEST(boundarySurfaces.at(3)->center() == pos, tt::tolerance(10e-12));
    BOOST_TEST(boundarySurfaces.at(2)->center() == pos, tt::tolerance(10e-12));
  }

}  // end of namespace Test

}  // end of namespace Acts
