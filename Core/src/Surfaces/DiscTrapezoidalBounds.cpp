// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 Acts project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// DiscTrapezoidalBounds.cpp, Acts project
///////////////////////////////////////////////////////////////////

#include "Acts/Surfaces/DiscTrapezoidalBounds.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

#include "Acts/Utilities/detail/periodic.hpp"

Acts::DiscTrapezoidalBounds::DiscTrapezoidalBounds(double minhalfx,
                                                   double maxhalfx,
                                                   double maxR,
                                                   double minR,
                                                   double avephi,
                                                   double stereo)
  : m_rMin(std::min(std::abs(minR), std::abs(maxR)))
  , m_rMax(std::max(std::abs(minR), std::abs(maxR)))
  , m_minHalfX(std::abs(minhalfx))
  , m_maxHalfX(std::abs(maxhalfx))
  , m_avgPhi(detail::radian_sym(avephi))
  , m_stereo(stereo)
{
}

Acts::DiscTrapezoidalBounds::~DiscTrapezoidalBounds() = default;

Acts::DiscTrapezoidalBounds*
Acts::DiscTrapezoidalBounds::clone() const
{
  return new DiscTrapezoidalBounds(*this);
}

Acts::SurfaceBounds::BoundsType
Acts::DiscTrapezoidalBounds::type() const
{
  return SurfaceBounds::DiscTrapezoidal;
}

std::vector<TDD_real_t>
Acts::DiscTrapezoidalBounds::valueStore() const
{
  std::vector<TDD_real_t> values(DiscTrapezoidalBounds::bv_length);
  values[bv_rMin]       = rMin();
  values[bv_rMax]       = rMax();
  values[bv_minHalfX]   = minHalflengthX();
  values[bv_maxHalfX]   = maxHalflengthX();
  values[bv_averagePhi] = averagePhi();
  values[bv_stereo]     = m_stereo;
  return values;
}

Acts::Vector2D
Acts::DiscTrapezoidalBounds::toLocalXY(const Acts::Vector2D& lpos) const
{
  return {lpos[eLOC_R] * std::sin(lpos[eLOC_PHI] - m_avgPhi),
          lpos[eLOC_R] * std::cos(lpos[eLOC_PHI] - m_avgPhi)};
}

Acts::ActsMatrixD<2, 2>
Acts::DiscTrapezoidalBounds::jacobianToLocalXY(const Acts::Vector2D& lpos) const
{
  ActsMatrixD<2, 2> jacobian;
  jacobian(0, eLOC_R)   = std::sin(lpos[eLOC_PHI] - m_avgPhi);
  jacobian(1, eLOC_R)   = std::cos(lpos[eLOC_PHI] - m_avgPhi);
  jacobian(0, eLOC_PHI) = lpos[eLOC_R] * std::cos(lpos[eLOC_PHI]);
  jacobian(1, eLOC_PHI) = lpos[eLOC_R] * -std::sin(lpos[eLOC_PHI]);
  return jacobian;
}

bool
Acts::DiscTrapezoidalBounds::inside(const Acts::Vector2D&      lpos,
                                    const Acts::BoundaryCheck& bcheck) const
{
  Vector2D vertices[] = {{minHalflengthX(), rMin()},
                         {maxHalflengthX(), rMax()},
                         {-maxHalflengthX(), rMax()},
                         {-minHalflengthX(), rMin()}};
  auto jacobian = jacobianToLocalXY(lpos);
  return bcheck.transformed(jacobian).isInside(toLocalXY(lpos), vertices);
}

double
Acts::DiscTrapezoidalBounds::distanceToBoundary(
    const Acts::Vector2D& lpos) const
{
  Vector2D vertices[] = {{minHalflengthX(), rMin()},
                         {maxHalflengthX(), rMax()},
                         {-maxHalflengthX(), rMax()},
                         {-minHalflengthX(), rMin()}};
  return BoundaryCheck(true).distance(toLocalXY(lpos), vertices);
}

// ostream operator overload
std::ostream&
Acts::DiscTrapezoidalBounds::toStream(std::ostream& sl) const
{
  sl << std::setiosflags(std::ios::fixed);
  sl << std::setprecision(5);
  sl << "Acts::DiscTrapezoidalBounds:  (innerRadius, outerRadius, hMinX, "
        "hMaxX, hlengthY, hPhiSector, averagePhi, rCenter, stereo) = ";
  sl << "(" << rMin() << ", " << rMax() << ", " << minHalflengthX() << ", "
     << maxHalflengthX() << ", " << halflengthY() << ", " << halfPhiSector()
     << ", " << averagePhi() << ", " << rCenter() << ", " << stereo() << ")";
  sl << std::setprecision(-1);
  return sl;
}
