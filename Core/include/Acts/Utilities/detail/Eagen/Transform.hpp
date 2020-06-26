// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <type_traits>
#include <utility>

#include "Block.hpp"
#include "DiagonalBase.hpp"
#include "EigenBase.hpp"
#include "EigenDense.hpp"
#include "EigenPrologue.hpp"
#include "RotationBase.hpp"
#include "Matrix.hpp"
#include "MatrixBase.hpp"

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
    // === Eagen wrapper API ===

    // Expose template parameters
    using Scalar = _Scalar;
    static constexpr int Dim = _Dim;
    static constexpr int Mode = _Mode;
    static constexpr int Options = _Options;

    // Wrapped Eigen type
    using Inner = Eigen::Transform<Scalar, Dim, Mode, Options>;

    // Access the wrapped Eigen object
    Inner& getInner() {
        return m_inner;
    }
    const Inner& getInner() const {
        return m_inner;
    }
    Inner&& moveInner() {
        return std::move(m_inner);
    }

    // === Eigen::Transform API ===

    // Eigen-style typedefs and consts
    using Index = Eigen::Index;
private:
    using ScalarTraits = NumTraits<Scalar>;
public:
    using RealScalar = typename ScalarTraits::Real;

    // Default constructor
    Transform() = default;

    // Constructor from inner type
    Transform(const Inner& inner)
        : m_inner(inner)
    {}

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
    template <typename OtherScalarType, int OtherMode, int OtherOptions>
    Transform(const Transform<OtherScalarType,
                              Dim,
                              OtherMode,
                              OtherOptions>& other)
        : m_inner(other.getInner())
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
    Transform
    inverse(TransformTraits traits = static_cast<TransformTraits>(Mode)) const {
        return Transform(m_inner.inverse(traits));
    }

    // Approximate equality
    bool isApprox(const Transform& other,
                  const RealScalar& prec = dummy_precision()) const {
        return m_inner.isApprox(other.getInner(), prec);
    }

    // Access the inner transformation matrix
    using MatrixType = Matrix<Scalar, Inner::Rows, Inner::HDim, Options>;
private:
    using MatrixTypeInner = typename MatrixType::Inner;
public:
    const MatrixType& matrix() const {
        const auto& resultInner = m_inner.matrix();
        static_assert(
            std::is_same_v<decltype(resultInner), const MatrixTypeInner&>,
            "Unexpected return type from Eigen in-place accessor");
        return reinterpret_cast<const MatrixType&>(resultInner);
    }
    MatrixType& matrix() {
        auto& resultInner = m_inner.matrix();
        static_assert(
            std::is_same_v<decltype(resultInner), MatrixTypeInner&>,
            "Unexpected return type from Eigen in-place accessor");
        return reinterpret_cast<MatrixType&>(resultInner);
    }

    // Access the linear part of the transform
private:
    static constexpr int NotRowMajor = ((Options & RowMajor) == 0);
    static constexpr bool LinearPartInnerPanel =
        (int(Mode) == AffineCompact) && NotRowMajor;
    template <typename _MatrixType>
    using GenericLinearPart = Block<_MatrixType,
                                    Dim,
                                    Dim,
                                    LinearPartInnerPanel>;
public:
    using ConstLinearPart = GenericLinearPart<const MatrixType>;
    ConstLinearPart linear() const {
        return ConstLinearPart(matrix(), 0, 0);
    }
    using LinearPart = GenericLinearPart<MatrixType>;
    LinearPart linear() {
        return LinearPart(matrix(), 0, 0);
    }

    // Access the rotation part of the transform
    using LinearMatrixType = Matrix<Scalar, Dim, Dim>;
    LinearMatrixType rotation() const {
        return m_inner.rotation();
    }

    // Access the translation part of the transform
private:
    template <typename _MatrixType>
    using GenericTranslationPart = Block<_MatrixType,
                                         Dim,
                                         1,
                                         NotRowMajor>;
public:
    using ConstTranslationPart = GenericTranslationPart<const MatrixType>;
    ConstTranslationPart translation() const {
        return ConstTranslationPart(matrix(), 0, Dim);
    }
    using TranslationPart = GenericTranslationPart<MatrixType>;
    TranslationPart translation() {
        return TranslationPart(matrix(), 0, Dim);
    }

    // Set the last row to [0 ... 0 1]
    void makeAffine() {
        m_inner.makeAffine();
    }

    // Coefficient accessors
    Scalar& operator()(Index row, Index col) {
        return m_inner(row, col);
    }
    Scalar operator()(Index row, Index col) const {
        return m_inner(row, col);
    }

    // Apply the transform to a matrix
    template <typename DiagonalDerived>
    Transform operator*(const DiagonalBase<DiagonalDerived>& b) const {
        return Transform(m_inner * b.derivedInner());
    }
    // NOTE: No Eigen::EigenBase overload needed, this is only ever applied to
    //       matrices, not arrays.
    template <typename OtherDerived>
    typename OtherDerived::PlainObject
    operator*(const EigenBase<OtherDerived>& other) const {
        return typename OtherDerived::PlainObject(
            m_inner * other.derivedInner()
        );
    }

    // Multiply two transforms with each other
private:
    static constexpr int TransformProductResultMode(int OtherMode) {
        return (Mode == (int)Projective
                || OtherMode == (int)Projective) ? Projective :
               (Mode == (int)Affine
                || OtherMode == (int)Affine) ? Affine :
               (Mode == (int)AffineCompact
                || OtherMode == (int)AffineCompact) ? AffineCompact :
               (Mode == (int)Isometry
                || OtherMode == (int)Isometry) ? Isometry :
               Projective;
    }
    template <int OtherMode>
    using TransformProductResult =
        Transform<Scalar,
                  Dim,
                  TransformProductResultMode(OtherMode),
                  Options>;
public:
    template <int OtherMode, int OtherOptions>
    TransformProductResult<OtherMode> operator*(
        const Transform<Scalar, Dim, OtherMode, OtherOptions>& other
    ) const {
        return TransformProductResult<OtherMode>(
            m_inner * other.getInner()
        );
    }

    // Translation interoperability
    using TranslationType = Translation<Scalar, Dim>;
private:
    using TranslationTypeInner = typename TranslationType::Inner;
public:
    Transform(const TranslationType& t)
        : m_inner(t.getInner())
    {}
    Transform(const TranslationTypeInner& t)
        : m_inner(t)
    {}
    Transform operator*(const TranslationType& t) const {
        return Transform(m_inner * t.getInner());
    }
    Transform& operator*=(const TranslationType& t) {
        m_inner *= t.getInner();
        return *this;
    }
    Transform& operator=(const Translation<Scalar, Dim>& t) {
        m_inner = t.getInner();
        return *this;
    }

    // Rotation interoperability
    template <typename OtherDerived>
    Transform(const RotationBase<OtherDerived, Dim>& r)
        : m_inner(r.derivedInner())
    {}
    template <typename OtherDerived>
    Transform operator*(const RotationBase<OtherDerived, Dim>& r) const {
        return Transform(m_inner * r.derivedInner());
    }
    template <typename OtherDerived>
    Transform& operator*=(const RotationBase<OtherDerived, Dim>& r) {
        m_inner *= r.derivedInner();
        return *this;
    }
    template <typename OtherDerived>
    Transform& operator=(const RotationBase<OtherDerived, Dim>& r) {
        m_inner = r.derivedInner();
        return *this;
    }

    // Assign an Eigen object (presumably a transform matrix)
    template <typename OtherDerived>
    Transform& operator=(const EigenBase<OtherDerived>& other) {
        m_inner = other.derivedInner();
        return *this;
    }
    template <typename OtherDerived>
    Transform& operator=(const Eigen::EigenBase<OtherDerived>& other) {
        m_inner = other;
        return *this;
    }

    // Pre-apply various transforms
    template <typename RotationType>
    Transform& prerotate(const RotationType& rotation) {
        m_inner.prerotate(rotation.derivedInner());
        return *this;
    }
    template <typename OtherDerived>
    Transform& prescale(const MatrixBase<OtherDerived>& other) {
        m_inner.prescale(other.derivedInner());
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

    // Post-apply various transforms
    template <typename RotationType>
    Transform& rotate(const RotationType& rotation) {
        m_inner.rotate(rotation.derivedInner());
        return *this;
    }
    template <typename OtherDerived>
    Transform& scale(const MatrixBase<OtherDerived>& other) {
        m_inner.scale(other.derivedInner());
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

    // Set to the identity transform
    void setIdentity() {
        m_inner.setIdentity();
    }

    // Build an identity transform
    static Transform Identity() {
        return Transform(Inner::Identity());
    }

    // Left-side multiplication
    template <typename DiagonalDerived>
    friend Transform operator*(const DiagonalBase<DiagonalDerived>& a,
                               const Transform& b) {
        return Transform(a.derivedInner() * b.getInner());
    }
    // TODO: Support left-side transform multiplication

private:
    Inner m_inner;

    static RealScalar dummy_precision() {
        return ScalarTraits::dummy_precision();
    }
};

using Affine2f = Transform<float, 2, Affine>;
using Affine3f = Transform<float, 3, Affine>;
using Affine2d = Transform<double, 2, Affine>;
using Affine3d = Transform<double, 3, Affine>;
using AffineCompact2f = Transform<float, 2, AffineCompact>;
using AffineCompact3f = Transform<float, 3, AffineCompact>;
using AffineCompact2d = Transform<double, 2, AffineCompact>;
using AffineCompact3d = Transform<double, 3, AffineCompact>;
using Isometry2f = Transform<float, 2, Isometry>;
using Isometry3f = Transform<float, 3, Isometry>;
using Isometry2d = Transform<double, 2, Isometry>;
using Isometry3d = Transform<double, 3, Isometry>;
using Projective2f = Transform<float, 2, Projective>;
using Projective3f = Transform<float, 3, Projective>;
using Projective2d = Transform<double, 2, Projective>;
using Projective3d = Transform<double, 3, Projective>;

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
