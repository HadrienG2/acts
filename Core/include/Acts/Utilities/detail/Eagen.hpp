// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <utility>

// for GNU: ignore this specific warning, otherwise just include Eigen/Dense
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include <Eigen/Dense>
#pragma GCC diagnostic pop
#else
#include <Eigen/Dense>
#endif

namespace Acts {

namespace detail {

/// Eagerly evaluated variant of Eigen, without expression templates
namespace Eagen {

// Propagate some Eigen types and constants
using Index = Eigen::Index;
using NoChange_t = Eigen::NoChange_t;
constexpr int AutoAlign = Eigen::AutoAlign;
constexpr int RowMajor = Eigen::RowMajor;
constexpr int ColMajor = Eigen::ColMajor;

// Forward declarations
template <typename Scalar, int Rows, int Cols,
          int Options = Eigen::AutoAlign |
                        ( (Rows==1 && Cols!=1) ? Eigen::RowMajor
                          : (Cols==1 && Rows!=1) ? Eigen::ColMajor
                          : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
          int MaxRows = Rows,
          int MaxCols = Cols>
class Matrix;
template <typename Derived> class PlainObjectBase;

// We don't need to replicate Eigen's full class hierarchy for now, but let's
// keep the useful metadata from Eigen's method signatures around
template <typename Derived> using DenseBase = PlainObjectBase<Derived>;
template <typename Derived> using DenseCoeffsBaseDW = PlainObjectBase<Derived>;
template <typename Derived> using DenseCoeffsBaseW = PlainObjectBase<Derived>;
template <typename Derived> using DenseCoeffsBaseRO = PlainObjectBase<Derived>;
template <typename Derived> using EigenBase = PlainObjectBase<Derived>;

// Spiritual equivalent of Eigen::PlainObjectBase and lower layers
//
// (We need to replicate that layer of the Eigen class hierarchy to support both
// Matrix and Array wrappers without code duplication.)
//
template <typename Derived>
class PlainObjectBase {
private:
    // Eigen type wrapped by the CRTP daughter class
    using Inner = typename Derived::Inner;

    // Template parameters of derived class
    using Scalar = typename Derived::Scalar;
    static constexpr int Rows = Derived::Rows;
    static constexpr int Cols = Derived::Cols;
    static constexpr int Options = Derived::Options;
    static constexpr int MaxRows = Derived::MaxRows;
    static constexpr int MaxCols = Derived::MaxCols;

public:
    // === Eigen::PlainObjectBase interface ===

    // Coefficient access
    template <typename... Index>
    const Scalar& coeff(Index... indices) const {
        return derivedInner().coeff(indices...);
    }
    template <typename... Index>
    const Scalar& coeffRef(Index... indices) const {
        return derivedInner().coeffRef(indices...);
    }
    template <typename... Index>
    Scalar& coeffRef(Index... indices) {
        return derivedInner().coeffRef(indices...);
    }

    // Resizing
    template <typename... ArgTypes>
    void conservativeResize(ArgTypes... args) {
        derivedInner().conservativeResize(args...);
    }
    template <typename OtherDerived>
    void conservativeResizeLike(const DenseBase<OtherDerived>& other) {
        derivedInner().conservativeResizeLike(other.derivedInner());
    }
    template <typename OtherDerived>
    void conservativeResizeLike(const Eigen::DenseBase<OtherDerived>& other) {
        derivedInner().conservativeResizeLike(other);
    }
    template <typename... ArgTypes>
    void resize(ArgTypes... args) {
        derivedInner().resize(args...);
    }
    template <typename OtherDerived>
    void resizeLike(const EigenBase<OtherDerived>& other) {
        derivedInner().resizeLike(other.derivedInner());
    }
    template <typename OtherDerived>
    void resizeLike(const Eigen::EigenBase<OtherDerived>& other) {
        derivedInner().resizeLike(other);
    }

    // Data access
    Scalar* data() { return derivedInner().data(); }
    const Scalar* data() const { return derivedInner().data(); }

    // Lazy assignment (?)
    template <typename OtherDerived>
    Derived& lazyAssign(const DenseBase<OtherDerived>& other) {
        derivedInner().lazyAssign(other.derivedInner());
        return *this;
    }
    template <typename OtherDerived>
    Derived& lazyAssign(const Eigen::DenseBase<OtherDerived>& other) {
        derivedInner().lazyAssign(other);
        return *this;
    }

    // Set inner values from various scalar sources
    template <typename... ArgTypes>
    Derived& setConstant(ArgTypes&&... args) {
        derivedInner().setConstant(std::forward(args)...);
        return *this;
    }
    template <typename... Index>
    Derived& setZero(Index... indices) {
        derivedInner().setZero(indices...);
        return *this;
    }
    template <typename... Index>
    Derived& setOnes(Index... indices) {
        derivedInner().setOnes(indices...);
        return *this;
    }
    template <typename... Index>
    Derived& setRandom(Index... indices) {
        derivedInner().setRandom(indices...);
        return *this;
    }

    // Map foreign data
    // TODO: Consider providing a first-class Map type to avoid copies
    template <typename... ArgTypes>
    static Derived Map(ArgTypes&&... args) {
        return Derived(Inner::Map(std::forward(args)...));
    }
    template <typename... ArgTypes>
    static Derived MapAligned(ArgTypes&&... args) {
        return Derived(Inner::MapAligned(std::forward(args)...));
    }

    // TODO: Replicate interface of Eigen::DenseBase
    // TODO: Replicate interface of all Eigen::DenseCoeffsBase types
    // TODO: Replicate interface of Eigen::EigenBase

protected:
    // Access the inner object held by the CRTP daughter class
    Inner& derivedInner() {
        return derived().getInner();
    }
    const Inner& derivedInner() const {
        return derived().getInner();
    }
    Inner&& moveDerivedInner() {
        return derived().moveInner();
    }

private:
    // Access the CRTP daughter class
    Derived& derived() {
        return *static_cast<Derived*>(this);
    }
    const Derived& derived() const {
        return *static_cast<const Derived*>(this);
    }
};

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

    // Re-expose template parameters
    using Scalar = _Scalar;
    static constexpr int Rows = _Rows;
    static constexpr int Cols = _Cols;
    static constexpr int Options = _Options;
    static constexpr int MaxRows = _MaxRows;
    static constexpr int MaxCols = _MaxCols;

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
    template <typename... ArgTypes>
    Matrix(ArgTypes&&... args) : m_inner(std::forward(args)...) {}
    template <typename Other>
    Matrix& operator=(Other&& other) {
        m_inner = std::forward(other);
        return *this;
    }

    // Underlying Eigen matrix type (used for CRTP)
    using Inner = Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxRows>;

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

    // Emulate Eigen::Matrix's base class typedef
    using Base = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;

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
    template <typename... ArgTypes>
    void applyOnTheLeft(ArgTypes&&... args) {
        m_inner.applyOnTheLeft(std::forward(args)...);
    }
    template <typename OtherDerived>
    void applyOnTheLeft(const EigenBase<OtherDerived>& other) {
        m_inner.applyOnTheLeft(other.derivedInner());
    }
    template <typename... ArgTypes>
    void applyOnTheRight(ArgTypes&&... args) {
        m_inner.applyOnTheRight(std::forward(args)...);
    }
    template <typename OtherDerived>
    void applyOnTheRight(const EigenBase<OtherDerived>& other) {
        m_inner.applyOnTheRight(other.derivedInner());
    }

    // FIXME: Support array(), which is used by Acts (requires Array type)

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
    Matrix<Scalar, Rows, Cols> normalized() {
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
    Matrix<Scalar, Rows, Cols> stableNormalized() const {
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
    Matrix<Scalar, Rows, Cols>
    cross(const Eigen::MatrixBase<OtherDerived>& other) const {
        return Matrix<Scalar, Rows, Cols>(m_inner.cross(other));
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix<Scalar, Rows, Cols>
    cross(const Matrix<OtherScalar,
                       OtherRows,
                       OtherCols,
                       OtherOptions,
                       OtherMaxRows,
                       OtherMaxCols>& other) const {
        return Matrix<Scalar, Rows, Cols>(m_inner.cross(other.m_inner));
    }
    template <typename OtherDerived>
    Matrix<Scalar, Rows, Cols>
    cross3(const Eigen::MatrixBase<OtherDerived>& other) const {
        return Matrix<Scalar, Rows, Cols>(m_inner.cross3(other));
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix<Scalar, Rows, Cols>
    cross3(const Matrix<OtherScalar,
                       OtherRows,
                       OtherCols,
                       OtherOptions,
                       OtherMaxRows,
                       OtherMaxCols>& other) const {
        return Matrix<Scalar, Rows, Cols>(m_inner.cross3(other.m_inner));
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
    Matrix<Scalar, Rows, Cols> inverse() const {
        return Matrix<Scalar, Rows, Cols>(m_inner.inverse());
    }

    // Special matrix queries
    bool isDiagonal(const RealScalar& prec = s_dummy_precision) const {
        return m_inner.isDiagonal(prec);
    }
    bool isIdentity(const RealScalar& prec = s_dummy_precision) const {
        return m_inner.isIdentity(prec);
    }
    bool isLowerTriangular(const RealScalar& prec = s_dummy_precision) const {
        return m_inner.isLowerTriangular(prec);
    }
    bool isUnitary(const RealScalar& prec = s_dummy_precision) const {
        return m_inner.isUnitary(prec);
    }
    bool isUpperTriangular(const RealScalar& prec = s_dummy_precision) const {
        return m_inner.isUpperTriangular(prec);
    }

    // TODO: Support lazyProduct()
    //       This is not currently used by Acts, so lower-priority.

    // FIXME: Support Cholesky decompositions (LLT, LDLT), used by Acts

    // NOTE: noalias() will probably never be supported, as it should almost
    //       never be needed in an eagerly evaluated world, where AFAIK its only
    //       use is to make operator*= slightly more efficient.

    // Operators
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
    template <typename OtherEigen>
    Matrix<Scalar, Rows, OtherEigen::ColsAtCompileTime>
    operator*(const OtherEigen& other) const {
        return Matrix<Scalar, Rows, OtherEigen::ColsAtCompileTime>(
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

private:
    Inner m_inner;

    static const RealScalar s_dummy_precision =
        Eigen::NumTraits<Scalar>::dummy_precision();
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
