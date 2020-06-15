// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EigenDense.hpp"
#include "EigenPrologue.hpp"
#include "Map.hpp"
#include "Matrix.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Wrapper of Eigen::Transform
//
// NOTE: QTransform interoperability is currently not supported, probably never
//       will be supported unless a need for it appears.
//
template <typename _Scalar,
          int _Dim,
          int _Mode,
          int _Options = AutoAlign>
class Transform {
public:
    // Expose template parameters
    using Scalar = _Scalar;
    static constexpr int Dim = _Dim;
    static constexpr int Mode = _Mode;
    static constexpr int Options = _Options;

    // Inner Eigen type
    using Inner = Eigen::Transform<Scalar, Dim, Mode, Options>;

    // Access the inner Eigen type
    Inner& getInner() {
        return m_inner;
    }
    const Inner& getInner() const {
        return m_inner;
    }

    // Eigen-style typedefs and consts
    using Index = Eigen::Index;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

    // Default constructor
    Transform() = default;

    // Constructor from a transformation matrix
    template <typename OtherDerived>
    explicit Transform(const EigenBase<OtherDerived>& other)
        : m_inner(other.derivedInner())
    {}
    template <typename OtherDerived>
    explicit Transform(const Eigen::EigenBase<OtherDerived>& other)
        : m_inner(other)
    {}

    // Constructor from another transform
    template <typename OtherScalarType>
    Transform(const Transform<OtherScalarType, Dim, Mode, Options>& other)
        : m_inner(other.m_inner)
    {}
    template <typename OtherScalarType>
    Transform(const Eigen::Transform<OtherScalarType, Dim, Mode, Options>& other)
        : m_inner(other)
    {}

    // TODO: Support affine()
    //       This requires quite a bit of work, and isn't used by Acts yet.

    // Cast to a different scalar type
    template <typename NewScalarType>
    Transform<NewScalarType, Dim, Mode, Options> cast() const {
        return Transform<NewScalarType, Dim, Mode, Options>(
            m_inner.template cast<NewScalarType>()
        );
    }

    // TODO: Support computeRotationScaling() and computeScalingRotation()
    //       This requires quite a bit of work, and isn't used by Acts yet.

    // Access inner data
    Scalar* data() {
        return m_inner.data();
    }
    const Scalar* data() const {
        return m_inner.data();
    }

    // TODO: Support fromPositionOrientationScale()
    //       This requires quite a bit of work, and isn't used by Acts yet.

    // Invert the transform
    Transform inverse(TransformTraits traits = static_cast<TransformTraits>(Mode)) const {
        return Transform(m_inner.inverse(traits));
    }

    // Approximate equality
    bool isApprox(const Transform& other,
                  const RealScalar& prec = dummy_precision()) const {
        return m_inner.isApprox(other.m_inner, prec);
    }
    bool isApprox(const Inner& other,
                  const RealScalar& prec = dummy_precision()) const {
        return m_inner.isApprox(other, prec);
    }

    // Access the linear part of the transform
private:
    using LinearPart = Matrix<Scalar, Dim, Dim>;
    using LinearPartMap = Map<LinearPart,
                              Unaligned,
                              Stride<Dim+1, 1>>;
public:
    LinearPart linear() const {
        return LinearPart(m_inner.linear());
    }
    LinearPartMap linear() {
        return LinearPartMap(m_inner.linear().data());
    }

    // Access the rotation part of the transform
    Matrix<Scalar, Dim, Dim> rotation() const {
        return m_inner.rotation();
    }

    // Access the translation part of the transform
private:
    using TranslationPart = Vector<Scalar, Dim>;
    using TranslationPartMap = Map<TranslationPart,
                                   Unaligned,
                                   Stride<Dim+1, Dim+1>>;
public:
    TranslationPart translation() const {
        return TranslationPart(m_inner.translation());
    }
    TranslationPartMap translation() {
        return TranslationPartMap(m_inner.translation().data());
    }

    // Set the last row to [0 ... 0 1]
    void makeAffine() {
        m_inner.makeAffine();
    }

    // Access the inner transformation matrix
private:
    using MatrixType = Matrix<Scalar, Inner::Rows, Inner::HDim, Options>;
    using MatrixTypeMap = Map<MatrixType>;
public:
    MatrixType matrix() const {
        return MatrixType(m_inner.matrix());
    }
    MatrixTypeMap matrix() {
        return MatrixTypeMap(m_inner.matrix().data());
    }

    // Coefficient accessors
    Scalar& operator()(Index row, Index col) {
        return m_inner(row, col);
    }
    Scalar operator()(Index row, Index col) const {
        return m_inner(row, col);
    }

    // Apply the transform to some Eigen object
    // NOTE: No Eagen-style Diagonal for now
    template <typename DiagonalDerived>
    Matrix<Scalar, Dim+1, Dim+1> operator*(const Eigen::DiagonalBase<DiagonalDerived>& b) const {
        return Matrix<Scalar, Dim+1, Dim+1>(m_inner * b);
    }
    template <typename OtherDerived>
    OtherDerived operator*(const EigenBase<OtherDerived>& other) const {
        return OtherDerived(m_inner * other.derivedInner());
    }
    // TODO: Support transforming Eigen::EigenBase too, requires figuring out
    //       the right Eagen matrix type.
    template <int OtherMode, int OtherOptions>
    Transform operator*(
        const Transform<Scalar, Dim, OtherMode, OtherOptions>& other
    ) const {
        return Transform(m_inner * other.m_inner);
    }

    // Assign an Eigen object (presumably a transform matrix)
    template <typename OtherDerived>
    Transform& operator=(const EigenBase<OtherDerived>& other) {
        m_inner = other.m_inner;
        return *this;
    }
    template <typename OtherDerived>
    Transform& operator=(const Eigen::EigenBase<OtherDerived>& other) {
        m_inner = other;
        return *this;
    }

    // Pre-apply various transforms
    // FIXME: Figure out how to integrate Eagen rotations here, once available
    template <typename RotationType>
    Transform& prerotate(const RotationType& rotation) {
        m_inner.prerotate(rotation);
        return *this;
    }
    template <typename OtherDerived>
    Transform& prescale(const MatrixBase<OtherDerived>& other) {
        m_inner.prescale(other.derivedInner());
        return *this;
    }
    template <typename OtherDerived>
    Transform& prescale(const Eigen::MatrixBase<OtherDerived>& other) {
        m_inner.prescale(other);
        return *this;
    }
    Transform& prescale(const Scalar& s) {
        m_inner.prescale(s);
        return *this;
    }
    Transform& preshear(const Scalar& sx, const Scalar& sy) {
        m_inner.preshear(sx, sy);
        return *this;
    }
    template <typename OtherDerived>
    Transform& pretranslate(const MatrixBase<OtherDerived>& other) {
        m_inner.pretranslate(other.derivedInner());
        return *this;
    }
    template <typename OtherDerived>
    Transform& pretranslate(const Eigen::MatrixBase<OtherDerived>& other) {
        m_inner.pretranslate(other);
        return *this;
    }

    // Post-apply various transforms
    // FIXME: Figure out how to integrate Eagen rotations here, once available
    template <typename RotationType>
    Transform& rotate(const RotationType& rotation) {
        m_inner.rotate(rotation);
        return *this;
    }
    template <typename OtherDerived>
    Transform& scale(const MatrixBase<OtherDerived>& other) {
        m_inner.scale(other.derivedInner());
        return *this;
    }
    template <typename OtherDerived>
    Transform& scale(const Eigen::MatrixBase<OtherDerived>& other) {
        m_inner.scale(other);
        return *this;
    }
    Transform& scale(const Scalar& s) {
        m_inner.scale(s);
        return *this;
    }
    Transform& shear(const Scalar& sx, const Scalar& sy) {
        m_inner.shear(sx, sy);
        return *this;
    }
    template <typename OtherDerived>
    Transform& translate(const MatrixBase<OtherDerived>& other) {
        m_inner.translate(other.derivedInner());
        return *this;
    }
    template <typename OtherDerived>
    Transform& translate(const Eigen::MatrixBase<OtherDerived>& other) {
        m_inner.translate(other);
        return *this;
    }

    // Set to the identity transform
    void setIdentity() {
        m_inner.setIdentity();
    }

    // Build an identity transform
    static Transform Identity() {
        return Transform(Inner::Identity());
    }

    // Left-side multiplication
    // NOTE: No diagonal support in Eagen yet
    template <typename DiagonalDerived>
    friend Transform operator*(const Eigen::DiagonalBase<DiagonalDerived>& a,
                               const Transform& b) {
        return Transform(a * b.m_inner);
    }
    // TODO: Support left-side transform multiplication

private:
    Inner m_inner;

    static RealScalar dummy_precision() {
        return Eigen::NumTraits<Scalar>::dummy_precision();
    }
};

using Affine2f = Transform<float,2,Affine>;
using Affine3f = Transform<float,3,Affine>;
using Affine2d = Transform<double,2,Affine>;
using Affine3d = Transform<double,3,Affine>;
using AffineCompact2f = Transform<float,2,AffineCompact>;
using AffineCompact3f = Transform<float,3,AffineCompact>;
using AffineCompact2d = Transform<double,2,AffineCompact>;
using AffineCompact3d = Transform<double,3,AffineCompact>;
using Isometry2f = Transform<float,2,Isometry>;
using Isometry3f = Transform<float,3,Isometry>;
using Isometry2d = Transform<double,2,Isometry>;
using Isometry3d = Transform<double,3,Isometry>;
using Projective2f = Transform<float,2,Projective>;
using Projective3f = Transform<float,3,Projective>;
using Projective2d = Transform<double,2,Projective>;
using Projective3d = Transform<double,3,Projective>;

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
