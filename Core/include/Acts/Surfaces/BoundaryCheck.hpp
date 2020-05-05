// This file is part of the Acts project.
//
// Copyright (C) 2016-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once
#include <cfloat>
#include <cmath>
#include <iterator>
#include <vector>

#include "Acts/Surfaces/detail/VerticesHelper.hpp"
#include "Acts/Utilities/Definitions.hpp"
#include "Acts/Utilities/ParameterDefinitions.hpp"

namespace Acts {

/// @class BoundaryCheck
///
/// The BoundaryCheck class provides boundary checks and distance calculations
/// for aligned box-like and polygonal boundaries on local surfaces.
/// Different types of boundary checks are supported and are transparently
/// selected when calling the `isInside(...)` and `distance(...)` methods:
///
/// -   Hard checks w/o any tolerances
/// -   Tolerance-based checks in one or in both local coordinates
/// -   Chi2-based checks based on a covariance matrix. Non-vanishing
///     correlations are correctly taken into account.
///
/// With a defined covariance matrix, the closest point and the distance are
/// not defined along the usual Euclidean metric, but by the Mahalanobis
/// distance induced by the the covariance.
class BoundaryCheck {
 public:
  /// Construct either hard cut in both dimensions or no cut at all.
  BoundaryCheck(bool check);

  /// Construct a tolerance based check.
  ///
  /// @param checkLocal0 Boolean directive to check coordinate 0
  /// @param checkLocal1 Boolean directive to check coordinate 1
  /// @param tolerance0 Tolerance along coordinate 0
  /// @param tolerance1 Tolerance along coordinate 1
  BoundaryCheck(bool checkLocal0, bool checkLocal1, double tolerance0 = 0,
                double tolerance1 = 0);

  /// Construct a chi2-based check.
  ///
  /// @param localCovariance Coverance matrix in local coordinates
  /// @param sigmaMax  Significance for the compatibility test
  BoundaryCheck(const ActsSymMatrixD<2>& localCovariance, double sigmaMax = 1);

  operator bool() const { return (m_type != Type::eNone); }
  bool operator!() const { return !bool(*this); }

  /// Check if the point is inside a polygon.
  ///
  /// @param point    Test point
  /// @param vertices Forward iterable container of convex polygon vertices.
  ///                 Calling `std::begin`/ `std::end` on the container must
  ///                 return an iterator where `*it` must be convertible to
  ///                 an `Acts::Vector2D`.
  ///
  /// The check takes into account whether tolerances or covariances are defined
  /// for the boundary check.
  template <typename Vector2DContainer>
  bool isInside(const Vector2D& point, const Vector2DContainer& vertices) const;

  /// Check if the point is inside a box aligned with the local axes.
  ///
  /// @param point   Test point
  /// @param loc0Min Lower bound along first axis
  /// @param loc0Max Upper bound along first axis
  /// @param loc1Min Lower bound along second axis
  /// @param loc1Max Upper bound along second axis
  ///
  /// The check takes into account whether tolerances or covariances are defined
  /// for the boundary check.
  bool isInside(const Vector2D& point, const Vector2D& lowerLeft,
                const Vector2D& upperRight) const;

  /// Calculate the signed, weighted, closest distance to a polygonal boundary.
  ///
  /// @param point Test point
  /// @param vertices Forward iterable container of convex polygon vertices.
  ///                 Calling `std::begin`/ `std::end` on the container must
  ///                 return an iterator where `*it` must be convertible to
  ///                 an `Acts::Vector2D`.
  /// @return Negative value if inside, positive if outside
  ///
  /// If a covariance is defined, the distance is the corresponding Mahalanobis
  /// distance. Otherwise, it is the Eucleadian distance.
  template <typename Vector2DContainer>
  double distance(const Vector2D& point,
                  const Vector2DContainer& vertices) const;

  /// Calculate the signed, weighted, closest distance to an aligned box.
  ///
  /// @param point Test point
  /// @param loc0Min Minimal value along the first local axis
  /// @param loc0Max Maximal value along the first local axis
  /// @param loc1Min Minimal value along the first local axis
  /// @param loc1Max Maximal value along the first local axis
  /// @return Negative value if inside, positive if outside
  ///
  /// If a covariance is defined, the distance is the corresponding Mahalanobis
  /// distance. Otherwise, it is the Eucleadian distance.
  double distance(const Vector2D& point, const Vector2D& lowerLeft,
                  const Vector2D& upperRight) const;

  enum class Type {
    eNone,      ///< disable boundary check
    eAbsolute,  ///< absolute cut
    eChi2       ///< chi2-based cut with full correlations
  };

  /// Broadcast the type
  Type type() const;

  // Broadcast the tolerance
  const Vector2D& tolerance() const;

  // Return the covariance matrix
  ActsSymMatrixD<2> covariance() const;

 private:
  /// Return a new BoundaryCheck with updated covariance.
  /// @param jacobian Tranform Jacobian for the covariance
  /// @warning This currently only transforms the covariance and does not work
  ///          for the tolerance based check.
  BoundaryCheck transformed(const ActsMatrixD<2, 2>& jacobian) const;

  /// Check if the distance vector is within the absolute or relative limits.
  bool isTolerated(const Vector2D& delta) const;

  /// Compute vector norm based on the covariance.
  double squaredNorm(const Vector2D& x) const;

  /// Calculate the closest point on the polygon.
  template <typename Vector2DContainer>
  Vector2D computeClosestPointOnPolygon(
      const Vector2D& point, const Vector2DContainer& vertices) const;

  /// Calculate the closest point on the box
  Vector2D computeEuclideanClosestPointOnRectangle(
      const Vector2D& point, const Vector2D& lowerLeft,
      const Vector2D& upperRight) const;

  /// metric weight matrix: identity for absolute mode or inverse covariance
  ActsSymMatrixD<2> m_weight;

  /// dual use: absolute tolerances or relative chi2/ sigma cut.
  Vector2D m_tolerance;
  Type m_type;

  // To acces the m_type
  friend class CylinderBounds;
  friend class RectangleBounds;
  // To be able to use `transformed`
  friend class CylinderBounds;
  friend class DiscTrapezoidBounds;
  // EllipseBounds needs a custom implementation
  friend class EllipseBounds;
};

}  // namespace Acts

inline Acts::BoundaryCheck::Type Acts::BoundaryCheck::type() const {
  return m_type;
}

inline const Acts::Vector2D& Acts::BoundaryCheck::tolerance() const {
  return m_tolerance;
}

inline Acts::ActsSymMatrixD<2> Acts::BoundaryCheck::covariance() const {
  return m_weight.inverse();
}

inline Acts::BoundaryCheck::BoundaryCheck(bool check)
    : m_weight(ActsSymMatrixD<2>::Identity()),
      m_tolerance(0, 0),
      m_type(check ? Type::eAbsolute : Type::eNone) {}

inline Acts::BoundaryCheck::BoundaryCheck(bool checkLocal0, bool checkLocal1,
                                          double tolerance0, double tolerance1)
    : m_weight(ActsSymMatrixD<2>::Identity()),
      m_tolerance(checkLocal0 ? tolerance0 : DBL_MAX,
                  checkLocal1 ? tolerance1 : DBL_MAX),
      m_type(Type::eAbsolute) {}

inline Acts::BoundaryCheck::BoundaryCheck(
    const ActsSymMatrixD<2>& localCovariance, double sigmaMax)
    : m_weight(localCovariance.inverse()),
      m_tolerance(sigmaMax, 0),
      m_type(Type::eChi2) {}

inline Acts::BoundaryCheck Acts::BoundaryCheck::transformed(
    const ActsMatrixD<2, 2>& jacobian) const {
  BoundaryCheck bc = *this;
  if (m_type == Type::eAbsolute) {
    // project tolerances to the new system. depending on the jacobian we need
    // to check both tolerances, even when the initial check does not.
    bc.m_tolerance = (jacobian * m_tolerance).cwiseAbs();
  } else /* Type::eChi2 */ {
    bc.m_weight =
        (jacobian * m_weight.inverse() * jacobian.transpose()).inverse();
  }
  return bc;
}

template <typename Vector2DContainer>
inline bool Acts::BoundaryCheck::isInside(
    const Vector2D& point, const Vector2DContainer& vertices) const {
  if (m_type == Type::eNone) {
    // The null boundary check always succeeds
    return true;
  } else if (detail::VerticesHelper::isInsidePolygon(point, vertices)) {
    // If the point falls inside the polygon, the check always succeeds
    return true;
  } else if (m_tolerance == Vector2D(0., 0.)) {
    // Outside of the polygon, since we've eliminated the case of an absence of
    // check above, we know we'll always fail if the tolerance is zero.
    //
    // This allows us to avoid the expensive computeClosestPointOnPolygon
    // computation in this simple case.
    //
    // TODO: When tolerance is not 0, we could also avoid this computation in
    //       some cases by testing against a bounding box of the polygon, padded
    //       on each side with our tolerance. Check if this optimization is
    //       worthwhile in some production workflows, and if so implement it.
    return false;
  } else {
    // We are outside of the polygon, but there is a tolerance. Must find what
    // the closest point on the polygon is and check if it's within tolerance.
    auto closestPoint = computeClosestPointOnPolygon(point, vertices);
    return isTolerated(closestPoint - point);
  }
}

inline bool Acts::BoundaryCheck::isInside(const Vector2D& point,
                                          const Vector2D& lowerLeft,
                                          const Vector2D& upperRight) const {
  if (detail::VerticesHelper::isInsideRectangle(point, lowerLeft, upperRight)) {
    return true;
  } else {
    Vector2D closestPoint;

    if (m_type == Type::eNone || m_type == Type::eAbsolute) {
      // absolute, can calculate directly
      closestPoint =
          computeEuclideanClosestPointOnRectangle(point, lowerLeft, upperRight);

    } else /* Type::eChi2 */ {
      // need to calculate by projection and squarednorm
      Vector2D vertices[] = {{lowerLeft[0], lowerLeft[1]},
                             {upperRight[0], lowerLeft[1]},
                             {upperRight[0], upperRight[1]},
                             {lowerLeft[0], upperRight[1]}};
      closestPoint = computeClosestPointOnPolygon(point, vertices);
    }

    return isTolerated(closestPoint - point);
  }
}

template <typename Vector2DContainer>
inline double Acts::BoundaryCheck::distance(
    const Acts::Vector2D& point, const Vector2DContainer& vertices) const {
  // TODO 2017-04-06 msmk: this should be calculable directly
  double d = squaredNorm(point - computeClosestPointOnPolygon(point, vertices));
  d = std::sqrt(d);
  return detail::VerticesHelper::isInsidePolygon(point, vertices) ? -d : d;
}

inline double Acts::BoundaryCheck::distance(const Acts::Vector2D& point,
                                            const Vector2D& lowerLeft,
                                            const Vector2D& upperRight) const {
  if (m_type == Type::eNone || m_type == Type::eAbsolute) {
    // compute closest point on box
    double d = (point - computeEuclideanClosestPointOnRectangle(
                            point, lowerLeft, upperRight))
                   .norm();
    return detail::VerticesHelper::isInsideRectangle(point, lowerLeft,
                                                     upperRight)
               ? -d
               : d;

  } else /* Type::eChi2 */ {
    Vector2D vertices[] = {{lowerLeft[0], lowerLeft[1]},
                           {upperRight[0], lowerLeft[1]},
                           {upperRight[0], upperRight[1]},
                           {lowerLeft[0], upperRight[1]}};
    return distance(point, vertices);
  }
}

inline bool Acts::BoundaryCheck::isTolerated(const Vector2D& delta) const {
  if (m_type == Type::eNone) {
    return true;
  } else if (m_type == Type::eAbsolute) {
    return (std::abs(delta[0]) <= m_tolerance[0]) &&
           (std::abs(delta[1]) <= m_tolerance[1]);
  } else /* Type::eChi2 */ {
    // Mahalanobis distances mean is 2 in 2-dim. cut is 1-d sigma.
    return (squaredNorm(delta) < (2 * m_tolerance[0]));
  }
}

inline double Acts::BoundaryCheck::squaredNorm(const Vector2D& x) const {
  return (x.transpose() * m_weight * x).value();
}

template <typename Vector2DContainer>
inline Acts::Vector2D Acts::BoundaryCheck::computeClosestPointOnPolygon(
    const Acts::Vector2D& point, const Vector2DContainer& vertices) const {
  // Our goal here is to find out what is the closest position on the input
  // polygon's edges,  as measured by the metric induced by the weight matrix.
  // To do this, we need to track what is the closest point that we observed so
  // far, and how close this point was.
  Vector2D closestPoint;
  double closestDistance = std::numeric_limits<double>::max();

  // Applying the weight matrix to a vector is costly, therefore we carefully
  // choose which vectors we apply it to, and cache the results, in order to
  // perform this multiplication as few times as possible.
  // - Pre-weight the current point
  const Vector2D weightedPoint = m_weight * point;
  // - Pre-weight the input vertices
  const size_t numVertices = std::size(vertices);
  Vector2D weightedVertices[numVertices];  // FIXME: VLAs are nonstandard & evil
  for (size_t i = 0; i < numVertices; ++i) {
    // FIXME: Indexed access to "vertices isn't guaranteed to work
    //        (present in various other places as well)
    weightedVertices[i] = m_weight * vertices[i];
  }

  // To study polygon edges, we keep track of the last vertex that we observed
  Vector2D prevVertex = vertices[numVertices-1];
  Vector2D weightedPrevVertex = weightedVertices[numVertices-1];

  // Experiment: Pre-compute raw u_l as well
  double u_l_raws[numEdges];
  for (size_t i = 0; i < numVertices; ++i) {
    const Vector2D currVertex = vertices[i];
    const Vector2D weightedCurrVertex = weightedVertices[i];
    const Vector2D n = currVertex - prevVertex;
    const Vector2D weightedN = weightedCurrVertex - weightedPrevVertex;
    const double f = n.dot(weightedN);
    u_l_raws[i] = (point - prevVertex).dot(weightedN) / f;
    prevVertex = currVertex;
    weightedPrevVertex = weightedCurrVertex;
  }

  // Experiment: Pre-compute u_s as well
  double u_ss[numEdges];
  for (size_t i = 0; i < numVertices; ++i) {
    const double u_l_raw = u_l_raws[i];
    const double u_l = std::isnormal(u_l_raw) ? u_l_raw : 0.5;
    u_ss[i] = std::clamp(u_l, 0.0, 1.0);
  }

  // Now, we're going to iterate over the remaining polygon vertices, and for
  // each of them we will study the closest point on the polygon edge defined by
  // the segment between the previously observed vertex and the current one.
  //
  // To process all polygon edges, we'll also need to consider the edge between
  // the last point of the polygon and the first one, which makes the iteration
  // pattern nontrivial. Hence we need to deduplicate the per-vertex logic in
  // this stateful lambda function.
  //
  auto processPolygonVertex = [&](size_t vertexIdx) {
    // Let's denote O the origin, V1 the previous vertex and V2 the current one
    // n is a vector going across the edge V1V2 that we're studying.
    const Vector2D currVertex = vertices[vertexIdx];

    // If you see through the caching of weighted vertices, this is just
    // f = (n.transpose() * m_weight * n).value() aka squaredNorm(n)
    const Vector2D weightedCurrVertex = weightedVertices[vertexIdx];

    // To get the closest point on this polygon edge, R, enforce that the
    // projection must lie on the segment between V1 and V2
    const double u_s = u_ss[vertexIdx];
    const Vector2D edgeClosest = (1.0 - u_s) * prevVertex + u_s * currVertex;

    // The distance is then defined by the vector PR = OR - OP.
    const Vector2D distVector = edgeClosest - point;

    // We compute the squared norm version of distVector using the following
    // optimized specialization of squaredNorm(), which avoids superfluous
    // m_weight application by reusing the weighted vectors we already have.
    const Vector2D weightedEdgeClosest =
      (1.0 - u_s) * weightedPrevVertex + u_s * weightedCurrVertex;
    const Vector2D weightedDistVector = weightedEdgeClosest - weightedPoint;
    const double edgeDistance = distVector.dot(weightedDistVector);

    // If this is smaller than the previously known smallest distance, the
    // closest point on this edge becomes our new estimate of the closest point
    // on all polygon edges.
    if (edgeDistance < closestDistance) {
      closestDistance = edgeDistance;
      closestPoint = edgeClosest;
    }

    // And finally, we prepare for the next edge of the polygon, which will
    // start at the current vertex.
    prevVertex = currVertex;
    weightedPrevVertex = weightedCurrVertex;
  };

  // Do the iteration over all polygon edges
  for(size_t i = 0; i < numVertices; ++i) {
    processPolygonVertex(i);
  }

  // TODO: Should be able to propagate closestDistance as well as closestPoint
  return closestPoint;
}

inline Acts::Vector2D
Acts::BoundaryCheck::computeEuclideanClosestPointOnRectangle(
    const Vector2D& point, const Vector2D& lowerLeft,
    const Vector2D& upperRight) const {
  /*
   *
   *        |                 |
   *   IV   |       V         | I
   *        |                 |
   *  ------------------------------
   *        |                 |
   *        |                 |
   *   VIII |     INSIDE      | VI
   *        |                 |
   *        |                 |
   *  ------------------------------
   *        |                 |
   *   III  |      VII        | II
   *        |                 |
   *
   */

  double l0 = point[0], l1 = point[1];
  double loc0Min = lowerLeft[0], loc0Max = upperRight[0];
  double loc1Min = lowerLeft[1], loc1Max = upperRight[1];

  // check if inside
  if (loc0Min <= l0 && l0 < loc0Max && loc1Min <= l1 && l1 < loc1Max) {
    // INSIDE
    double dist = std::abs(loc0Max - l0);
    Vector2D cls(loc0Max, l1);

    double test = std::abs(loc0Min - l0);
    if (test <= dist) {
      dist = test;
      cls = {loc0Min, l1};
    }

    test = std::abs(loc1Max - l1);
    if (test <= dist) {
      dist = test;
      cls = {l0, loc1Max};
    }

    test = std::abs(loc1Min - l1);
    if (test <= dist) {
      return {l0, loc1Min};
    }
    return cls;
  } else {
    // OUTSIDE, check sectors
    if (l0 > loc0Max) {
      if (l1 > loc1Max) {  // I
        return {loc0Max, loc1Max};
      } else if (l1 <= loc1Min) {  // II
        return {loc0Max, loc1Min};
      } else {  // VI
        return {loc0Max, l1};
      }
    } else if (l0 < loc0Min) {
      if (l1 > loc1Max) {  // IV
        return {loc0Min, loc1Max};
      } else if (l1 <= loc1Min) {  // III
        return {loc0Min, loc1Min};
      } else {  // VIII
        return {loc0Min, l1};
      }
    } else {
      if (l1 > loc1Max) {  // V
        return {l0, loc1Max};
      } else {  // l1 <= loc1Min # VII
        return {l0, loc1Min};
      }
      // third case not necessary, see INSIDE above
    }
  }
}
