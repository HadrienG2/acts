// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "EigenDense.hpp"
#include "ForwardDeclarations.hpp"
#include "PlainObjectBase.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Equivalent of Eigen's vector typedefs
template <typename Type, int Size>
using Vector = Matrix<Type, Size, 1>;
template <typename Type, int Size>
using RowVector = Matrix<Type, 1, Size>;

// Eigen::Matrix, but with eagerly evaluated operations
template <typename _Scalar,
          int _Rows,
          int _Cols,
          int _Options,
          int _MaxRows,
          int _MaxCols>
class Matrix : public PlainObjectBase<Matrix<_Scalar,
                                             _Rows,
                                             _Cols,
                                             _Options,
                                             _MaxRows,
                                             _MaxCols>> {
public:
    // TODO: If this works, reduce reliance on variadic templates by using the
    //       true method signatures instead.

    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Scalar = _Scalar;
    static constexpr int Rows = _Rows;
    static constexpr int Cols = _Cols;
    static constexpr int Options = _Options;
    static constexpr int MaxRows = _MaxRows;
    static constexpr int MaxCols = _MaxCols;

    // Re-expose Index typedef
    using Index = Eigen::Index;

    // Quick way to construct a matrix of the same size, but w/o the options
    using PlainBase = Matrix<Scalar, Rows, Cols>;

    // Underlying Eigen matrix type (used for CRTP)
    using Inner = Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;

    // Access the inner Eigen matrix (used for CRTP)
    Inner& getInner() {
        return m_inner;
    }
    const Inner& getInner() const {
        return m_inner;
    }
    Inner&& moveInner() {
        return std::move(m_inner);
    }

    // === Eigen::MatrixBase interface ===

    // Adjoint = conjugate + transpose
    Matrix<Scalar, Cols, Rows> adjoint() const {
        return Matrix<Scalar, Cols, Rows>(m_inner.adjoint());
    }
    void adjointInPlace() {
        m_inner.adjointInPlace();
    }

    // TODO: Support methods involving Householder transformations
    //       These are not currently used by Acts, so lower-priority.

    // In-place products
    template <typename... Args>
    void applyOnTheLeft(Args&&... args) {
        m_inner.applyOnTheLeft(std::forward<Args>(args)...);
    }
    template <typename OtherDerived>
    void applyOnTheLeft(const EigenBase<OtherDerived>& other) {
        m_inner.applyOnTheLeft(other.derivedInner());
    }
    template <typename... Args>
    void applyOnTheRight(Args&&... args) {
        m_inner.applyOnTheRight(std::forward<Args>(args)...);
    }
    template <typename OtherDerived>
    void applyOnTheRight(const EigenBase<OtherDerived>& other) {
        m_inner.applyOnTheRight(other.derivedInner());
    }

    // Switch to array view of this matrix
    //
    // FIXME: This currently leaves all the work to Eigen::Array, with the usual
    //        compile-time bloat problems that using Eigen types entails.
    //
    //        An eagerly evaluated Eagen::Array wrapper would be preferrable,
    //        but is very costly in terms of Eagen code complexity and
    //        development time (large API surface, need to replicate more Eigen
    //        metaprogramming sorcery for PlainObjectBase base class
    //        selection...), while the usage of Eigen::Array in the Acts
    //        codebase is actually very low.
    //
    //        So for now, I'm punting on this particular development.
    //
    Eigen::ArrayWrapper<Inner> array() {
        return m_inner.array();
    }
    Eigen::ArrayWrapper<const Inner> array() const {
        return m_inner.array();
    }

    // Reinterpret a vector as a diagonal matrix
    // TODO: Consider providing a dedicated diagonal wrapper type to avoid
    //       copies and extra run-time storage and computation overhead
    Matrix<Scalar, std::max(Rows, Cols), std::max(Rows, Cols)>
    asDiagonal() const {
        assert(std::min(Rows, Cols) == 1);
        return Matrix<Scalar, std::max(Rows, Cols), std::max(Rows, Cols)>(
            m_inner.asDiagonal()
        );
    }

    // TODO: Support SVD-related methods bdcSvd() and jacobiSvd()
    //       These are not currently used by Acts, so lower-priority.

    // Norms and normalization
    using RealScalar = typename Inner::RealScalar;
    RealScalar blueNorm() const {
        return m_inner.blueNorm();
    }
    // TODO: Support hnormalized()
    //       This is not currently used by Acts, so lower-priority.
    RealScalar hypotNorm() const {
        return m_inner.hypotNorm();
    }
    RealScalar lpNorm() const {
        return m_inner.lpNorm();
    }
    RealScalar norm() const {
        return m_inner.norm();
    }
    void normalize() {
        m_inner.normalize();
    }
    PlainBase normalized() {
        return m_inner.normalized();
    }
    RealScalar operatorNorm() const {
        return m_inner.operatorNorm();
    }
    RealScalar squaredNorm() const {
        return m_inner.squaredNorm();
    }
    RealScalar stableNorm() const {
        return m_inner.stableNorm();
    }
    void stableNormalize() {
        m_inner.stableNormalize();
    }
    PlainBase stableNormalized() const {
        return m_inner.stableNormalized();
    }

    // TODO: Support orthogonal matrix and decomposition features
    //       This is not currently used by Acts, so lower-priority.
    // TODO: Support computeInverse(AndDet)?WithCheck
    //       This is not currently used by Acts, so lower-priority.
    // TODO: Support Eigen's unsupported MatrixFunctions module
    //       This is not currently used by Acts, so lower-priority.

    // Cross products
    template <typename OtherDerived>
    PlainBase
    cross(const Eigen::MatrixBase<OtherDerived>& other) const {
        return PlainBase(m_inner.cross(other));
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    PlainBase
    cross(const Matrix<OtherScalar,
                       OtherRows,
                       OtherCols,
                       OtherOptions,
                       OtherMaxRows,
                       OtherMaxCols>& other) const {
        return PlainBase(m_inner.cross(other.m_inner));
    }
    template <typename OtherDerived>
    PlainBase
    cross3(const Eigen::MatrixBase<OtherDerived>& other) const {
        return PlainBase(m_inner.cross3(other));
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    PlainBase
    cross3(const Matrix<OtherScalar,
                       OtherRows,
                       OtherCols,
                       OtherOptions,
                       OtherMaxRows,
                       OtherMaxCols>& other) const {
        return PlainBase(m_inner.cross3(other.m_inner));
    }

    // Determinant
    Scalar determinant() const {
        return m_inner.determinant();
    }

    // Diagonal and associated properties
    // FIXME: Support non-const diagonal access, which is used by Acts for
    //        comma initialization based diagonal assignment.
    //        Use that to make the diagonal access method below more efficient.
    template <typename... Index>
    Vector<Scalar, Rows> diagonal(Index... indices) const {
        return Vector<Scalar, Rows>(m_inner.diagonal(indices...));
    }
    Index diagonalSize() const {
        return m_inner.diagonalSize();
    }

    // Dot product
    template <typename OtherDerived>
    Scalar dot(const Eigen::MatrixBase<OtherDerived>& other) const {
        return m_inner.dot(other);
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Scalar dot(const Matrix<OtherScalar,
                            OtherRows,
                            OtherCols,
                            OtherOptions,
                            OtherMaxRows,
                            OtherMaxCols>& other) const {
        return m_inner.dot(other.m_inner);
    }

    // Eigenvalues
    Vector<Scalar, Rows> eigenvalues() const {
        return Vector<Scalar, Rows>(m_inner.eigenvalues());
    }

    // Euler angles
    Vector<Scalar, 3> eulerAngles(Index a0, Index a1, Index a2) const {
        return Vector<Scalar, 3>(m_inner.eulerAngles());
    }

    // TODO: Support aligned access enforcement
    //       This is not currently used by Acts, so lower-priority.
    // TODO: Support LU decomposition
    //       This is not currently used by Acts, so lower-priority.
    // TODO: Support homogeneous()
    //       This is not currently used by Acts, so lower-priority.

    // Matrix inversion
    PlainBase inverse() const {
        return PlainBase(m_inner.inverse());
    }

    // Approximate comparisons
    bool isDiagonal(const RealScalar& prec = dummy_precision()) const {
        return m_inner.isDiagonal(prec);
    }
    bool isIdentity(const RealScalar& prec = dummy_precision()) const {
        return m_inner.isIdentity(prec);
    }
    bool isLowerTriangular(const RealScalar& prec = dummy_precision()) const {
        return m_inner.isLowerTriangular(prec);
    }
    bool isUnitary(const RealScalar& prec = dummy_precision()) const {
        return m_inner.isUnitary(prec);
    }
    bool isUpperTriangular(const RealScalar& prec = dummy_precision()) const {
        return m_inner.isUpperTriangular(prec);
    }

    // TODO: Support lazyProduct()
    //       This is not currently used by Acts, so lower-priority.

    // FIXME: Support Cholesky decompositions (LLT, LDLT), used by Acts

    // NOTE: noalias() will probably never be supported, as it should almost
    //       never be needed in an eagerly evaluated world, where AFAIK its only
    //       use is to make operator*= slightly more efficient.

    // Equality and inequality
    template <typename OtherDerived>
    bool operator==(const Eigen::MatrixBase<OtherDerived>& other) const {
        return (m_inner == other);
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    bool operator==(const Matrix<OtherScalar,
                                 OtherRows,
                                 OtherCols,
                                 OtherOptions,
                                 OtherMaxRows,
                                 OtherMaxCols>& other) const {
        return (m_inner == other.m_inner);
    }
    template <typename Other>
    bool operator!=(const Other& other) const {
        return !(*this == other);
    }

    // Matrix-scalar multiplication
    template <typename OtherScalar>
    PlainBase operator*(const OtherScalar& scalar) const {
        return PlainBase(m_inner * scalar);
    }
    template <typename OtherScalar>
    friend PlainBase operator*(const OtherScalar& scalar, const Matrix& matrix) {
        return matrix * scalar;
    }
    template <typename OtherScalar>
    Matrix& operator*=(const OtherScalar& scalar) {
        m_inner *= scalar;
        return *this;
    }

    // Matrix-scalar division
    template <typename OtherScalar>
    PlainBase operator/(const OtherScalar& scalar) const {
        return PlainBase(m_inner / scalar);
    }
    template <typename OtherScalar>
    Matrix& operator/=(const OtherScalar& scalar) {
        m_inner /= scalar;
        return *this;
    }

    // Matrix-matrix multiplication
    template <typename DiagonalDerived>
    PlainBase
    operator*(const Eigen::DiagonalBase<DiagonalDerived>& other) const {
        return PlainBase(m_inner * other);
    }
    template <typename OtherDerived>
    Matrix<Scalar, Rows, Eigen::MatrixBase<OtherDerived>::ColsAtCompileTime>
    operator*(const Eigen::MatrixBase<OtherDerived>& other) const {
        return Matrix<Scalar,
                      Rows,
                      Eigen::MatrixBase<OtherDerived>::ColsAtCompileTime>(
            m_inner * other
        );
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix<Scalar, Rows, OtherCols>
    operator*(const Matrix<OtherScalar,
                           OtherRows,
                           OtherCols,
                           OtherOptions,
                           OtherMaxRows,
                           OtherMaxCols>& other) const {
        return Matrix<Scalar, Rows, OtherCols>(
            m_inner * other.m_inner
        );
    }
    template <typename OtherDerived>
    Matrix& operator*=(const Eigen::EigenBase<OtherDerived>& other) {
        m_inner *= other;
        return *this;
    }
    template <typename OtherDerived>
    Matrix& operator*=(const EigenBase<OtherDerived>& other) {
        m_inner *= other.m_inner;
        return *this;
    }

    // Matrix addition
    template <typename OtherEigen>
    PlainBase
    operator+(const OtherEigen& other) const {
        return PlainBase(m_inner + other);
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    PlainBase
    operator+(const Matrix<OtherScalar,
                           OtherRows,
                           OtherCols,
                           OtherOptions,
                           OtherMaxRows,
                           OtherMaxCols>& other) const {
        return PlainBase(m_inner + other.m_inner);
    }
    template <typename OtherDerived>
    Matrix& operator+=(const Eigen::MatrixBase<OtherDerived>& other) {
        m_inner += other;
        return *this;
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix& operator+=(const Matrix<OtherScalar,
                                    OtherRows,
                                    OtherCols,
                                    OtherOptions,
                                    OtherMaxRows,
                                    OtherMaxCols>& other) {
        m_inner += other.m_inner;
        return *this;
    }

    // Matrix subtraction
    template <typename OtherEigen>
    PlainBase
    operator-(const OtherEigen& other) const {
        return PlainBase(m_inner + other);
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    PlainBase
    operator-(const Matrix<OtherScalar,
                           OtherRows,
                           OtherCols,
                           OtherOptions,
                           OtherMaxRows,
                           OtherMaxCols>& other) const {
        return PlainBase(m_inner + other.m_inner);
    }
    template <typename OtherDerived>
    Matrix& operator-=(const Eigen::MatrixBase<OtherDerived>& other) {
        m_inner -= other;
        return *this;
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix& operator-=(const Matrix<OtherScalar,
                                    OtherRows,
                                    OtherCols,
                                    OtherOptions,
                                    OtherMaxRows,
                                    OtherMaxCols>& other) {
        m_inner -= other.m_inner;
        return *this;
    }

    // TODO: Support selfadjointView()
    //       This is not currently used by Acts, so lower-priority.

    // Set to identity matrix
    template <typename... Index>
    Matrix& setIdentity(Index... indices) {
        m_inner.setIdentity(indices...);
        return *this;
    }

    // TODO: Support sparse matrices
    //       This is not currently used by Acts, so lower-priority.

    // Matrix trace
    Scalar trace() const {
        return m_inner.trace();
    }

    // TODO: Support triangular view
    //       This is not currently used by Acts, so lower-priority.

    // Coefficient-wise operations
    template <typename Other>
    PlainBase cwiseMin(const Other& other) const {
        return PlainBase(m_inner.cwiseMin(other));
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    PlainBase cwiseMin(const Matrix<OtherScalar,
                                    OtherRows,
                                    OtherCols,
                                    OtherOptions,
                                    OtherMaxRows,
                                    OtherMaxCols>& other) const {
        return PlainBase(m_inner.cwiseMin(other.m_inner));
    }
    template <typename Other>
    PlainBase cwiseMax(const Other& other) const {
        return PlainBase(m_inner.cwiseMax(other));
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    PlainBase cwiseMax(const Matrix<OtherScalar,
                                    OtherRows,
                                    OtherCols,
                                    OtherOptions,
                                    OtherMaxRows,
                                    OtherMaxCols>& other) const {
        return PlainBase(m_inner.cwiseMax(other.m_inner));
    }
    Matrix<RealScalar, Rows, Cols> cwiseAbs2() const {
        return Matrix<RealScalar, Rows, Cols>(m_inner.cwiseAbs2());
    }
    Matrix<RealScalar, Rows, Cols> cwiseAbs() const {
        return Matrix<RealScalar, Rows, Cols>(m_inner.cwiseAbs());
    }
    PlainBase cwiseSqrt() const {
        return PlainBase(m_inner.cwiseSqrt());
    }
    PlainBase cwiseInverse() const {
        return PlainBase(m_inner.cwiseInverse());
    }
    template <typename OtherDerived>
    PlainBase cwiseProduct(const Eigen::MatrixBase<OtherDerived>& other) const {
        return PlainBase(m_inner.cwiseProduct(other));
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    PlainBase cwiseProduct(const Matrix<OtherScalar,
                                        OtherRows,
                                        OtherCols,
                                        OtherOptions,
                                        OtherMaxRows,
                                        OtherMaxCols>& other) const {
        return PlainBase(m_inner.cwiseProduct(other.m_inner));
    }
    template <typename OtherDerived>
    PlainBase cwiseQuotient(const Eigen::MatrixBase<OtherDerived>& other) const {
        return PlainBase(m_inner.cwiseQuotient(other));
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    PlainBase cwiseQuotient(const Matrix<OtherScalar,
                                         OtherRows,
                                         OtherCols,
                                         OtherOptions,
                                         OtherMaxRows,
                                         OtherMaxCols>& other) const {
        return PlainBase(m_inner.cwiseQuotient(other.m_inner));
    }
    template <typename Other>
    PlainBase cwiseEqual(const Other& other) const {
        return PlainBase(m_inner.cwiseEqual(other));
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    PlainBase cwiseEqual(const Matrix<OtherScalar,
                                      OtherRows,
                                      OtherCols,
                                      OtherOptions,
                                      OtherMaxRows,
                                      OtherMaxCols>& other) const {
        return PlainBase(m_inner.cwiseEqual(other.m_inner));
    }
    template <typename OtherDerived>
    PlainBase cwiseNotEqual(const Eigen::MatrixBase<OtherDerived>& other) const {
        return PlainBase(m_inner.cwiseNotEqual(other));
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    PlainBase cwiseNotEqual(const Matrix<OtherScalar,
                                         OtherRows,
                                         OtherCols,
                                         OtherOptions,
                                         OtherMaxRows,
                                         OtherMaxCols>& other) const {
        return PlainBase(m_inner.cwiseNotEqual(other.m_inner));
    }

    // Special matrix generators
    template <typename... Index>
    static Matrix Identity(Index... indices) {
        return Matrix(Inner::Identity(indices...));
    }
    template <typename... Index>
    static Matrix Unit(Index... indices) {
        return Matrix(Inner::Unit(indices...));
    }
    static Matrix UnitW() {
        return Matrix(Inner::UnitW());
    }
    static Matrix UnitX() {
        return Matrix(Inner::UnitX());
    }
    static Matrix UnitY() {
        return Matrix(Inner::UnitY());
    }
    static Matrix UnitZ() {
        return Matrix(Inner::UnitZ());
    }

    // === Eigen::Matrix interface ===

    // Basic lifecycle
    Matrix() = default;
    template <typename OtherDerived>
    Matrix(const EigenBase<OtherDerived>& other)
        : m_inner(other.derivedInner())
    {}
    template <typename OtherDerived>
    Matrix& operator=(const EigenBase<OtherDerived>& other) {
        m_inner = other.derivedInner();
        return *this;
    }
#if EIGEN_HAS_RVALUE_REFERENCES
    template <typename OtherDerived>
    Matrix(EigenBase<OtherDerived>&& other)
        : m_inner(other.moveDerivedInner())
    {}
    template <typename OtherDerived>
    Matrix& operator=(EigenBase<OtherDerived>&& other) {
        m_inner = other.moveDerivedInner();
        return *this;
    }
#endif

    // Build and assign from anything Eigen supports building or assigning from
    template <typename... Args>
    Matrix(Args&&... args) : m_inner(std::forward<Args>(args)...) {}
    template <typename Other>
    Matrix& operator=(Other&& other) {
        m_inner = std::forward<Other>(other);
        return *this;
    }

    // Emulate Eigen::Matrix's base class typedef
    using Base = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;

private:
    Inner m_inner;

    static RealScalar dummy_precision() {
        return Eigen::NumTraits<Scalar>::dummy_precision();
    }
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
