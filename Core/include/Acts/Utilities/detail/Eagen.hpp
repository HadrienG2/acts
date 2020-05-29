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

// Forward declarations
template <typename Scalar, int Rows, int Cols,
          int Options = Eigen::AutoAlign |
                        ( (Rows==1 && Cols!=1) ? Eigen::RowMajor
                          : (Cols==1 && Rows!=1) ? Eigen::ColMajor
                          : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
          int MaxRows = Rows,
          int MaxCols = Cols>
class Matrix;

// Equivalent of Eigen's vector typedefs
template <typename Type, int Size>
using Vector = Matrix<Type, Size, 1>;
template <typename Type, int Size>
using RowVector = Matrix<Type, 1, Size>;

// Eigen::Matrix, but with eagerly evaluated operations
template <typename Scalar,
          int Rows,
          int Cols,
          int Options,
          int MaxRows,
          int MaxCols>
class Matrix {
private:
    using Inner = Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxRows>;

public:
    // TODO: If this works, reduce reliance on variadic templates by using the
    //       true method signatures instead.

    // === Eigen::Matrix interface ===

    // Basic lifecycle
    Matrix() = default;
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix(const Matrix<OtherScalar,
                        OtherRows,
                        OtherCols,
                        OtherOptions,
                        OtherMaxRows,
                        OtherMaxCols>& other)
        : m_inner(other.m_inner)
    {}
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix& operator=(const Matrix<OtherScalar,
                                   OtherRows,
                                   OtherCols,
                                   OtherOptions,
                                   OtherMaxRows,
                                   OtherMaxCols>& other) {
        m_inner = other.m_inner;
        return *this;
    }
#if EIGEN_HAS_RVALUE_REFERENCES
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix(Matrix<OtherScalar,
                  OtherRows,
                  OtherCols,
                  OtherOptions,
                  OtherMaxRows,
                  OtherMaxCols>&& other)
        : m_inner(std::move(other.m_inner))
    {}
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix& operator=(Matrix<OtherScalar,
                             OtherRows,
                             OtherCols,
                             OtherOptions,
                             OtherMaxRows,
                             OtherMaxCols>&& other) {
        m_inner = std::move(other.m_inner);
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

    // Access the inner Eigen matrix
    Inner& getEigen() {
        return m_inner;
    }
    const Inner& getEigen() const {
        return m_inner;
    }

    // Emulate Eigen::Matrix's base class typedef
    using Base = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;

    // === Eigen::PlainObjectBase interface ===

    // Coefficient access
    template <typename... Index>
    const Scalar& coeff(Index... indices) const {
        return m_inner.coeff(indices...);
    }
    template <typename... Index>
    const Scalar& coeffRef(Index... indices) const {
        return m_inner.coeffRef(indices...);
    }
    template <typename... Index>
    Scalar& coeffRef(Index... indices) {
        return m_inner.coeffRef(indices...);
    }

    // Resizing
    template <typename... ArgTypes>
    void conservativeResize(ArgTypes... args) {
        m_inner.conservativeResize(args...);
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    void conservativeResizeLike(const Matrix<OtherScalar,
                                             OtherRows,
                                             OtherCols,
                                             OtherOptions,
                                             OtherMaxRows,
                                             OtherMaxCols>& other) {
        m_inner.conservativeResizeLike(other.m_inner);
    }
    template <typename OtherDerived>
    void conservativeResizeLike(const Eigen::DenseBase<OtherDerived>& other) {
        m_inner.conservativeResizeLike(other);
    }
    template <typename... ArgTypes>
    void resize(ArgTypes... args) {
        m_inner.resize(args...);
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    void resizeLike(const Matrix<OtherScalar,
                                 OtherRows,
                                 OtherCols,
                                 OtherOptions,
                                 OtherMaxRows,
                                 OtherMaxCols>& other) {
        m_inner.resizeLike(other.m_inner);
    }
    template <typename OtherDerived>
    void resizeLike(const Eigen::DenseBase<OtherDerived>& other) {
        m_inner.resizeLike(other);
    }

    // Data access
    Scalar* data() { return m_inner.data(); }
    const Scalar* data() const { return m_inner.data(); }

    // Lazy assignment
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix& lazyAssign(const Matrix<OtherScalar,
                                    OtherRows,
                                    OtherCols,
                                    OtherOptions,
                                    OtherMaxRows,
                                    OtherMaxCols>& other) {
        m_inner.lazyAssign(other.m_inner);
        return *this;
    }
    template <typename OtherDerived>
    Matrix& lazyAssign(const Eigen::DenseBase<OtherDerived>& other) {
        m_inner.lazyAssign(other);
        return *this;
    }

    // Set inner values from various scalar sources
    template <typename... ArgTypes>
    Matrix& setConstant(ArgTypes&&... args) {
        m_inner.setConstant(std::forward(args)...);
        return *this;
    }
    template <typename... Index>
    Matrix& setZero(Index... indices) {
        m_inner.setZero(indices...);
        return *this;
    }
    template <typename... Index>
    Matrix& setOnes(Index... indices) {
        m_inner.setOnes(indices...);
        return *this;
    }
    template <typename... Index>
    Matrix& setRandom(Index... indices) {
        m_inner.setRandom(indices...);
        return *this;
    }

    // Map foreign data
    // TODO: Consider avoiding copies by providing a first-class Map type
    template <typename... ArgTypes>
    static Matrix Map(ArgTypes&&... args) {
        return Matrix(Inner::Map(std::forward(args)...));
    }
    template <typename... ArgTypes>
    static Matrix MapAligned(ArgTypes&&... args) {
        return Matrix(Inner::MapAligned(std::forward(args)...));
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
    template <typename... ArgTypes>
    void applyOnTheLeft(ArgTypes&&... args) {
        m_inner.applyOnTheLeft(std::forward(args)...);
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    void applyOnTheLeft(const Matrix<OtherScalar,
                                     OtherRows,
                                     OtherCols,
                                     OtherOptions,
                                     OtherMaxRows,
                                     OtherMaxCols>& other) {
        m_inner.applyOnTheLeft(other.m_inner);
    }
    template <typename... ArgTypes>
    void applyOnTheRight(ArgTypes&&... args) {
        m_inner.applyOnTheRight(std::forward(args)...);
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    void applyOnTheRight(const Matrix<OtherScalar,
                                      OtherRows,
                                      OtherCols,
                                      OtherOptions,
                                      OtherMaxRows,
                                      OtherMaxCols>& other) {
        m_inner.applyOnTheRight(other.m_inner);
    }

    // FIXME: Support array(), which is used by Acts (requires Array type)

    // Reinterpret a vector as a diagonal matrix
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

    // TODO: Support orthogonal matrix related features
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
    //        comma initialization based assignment.
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
    bool isDiagonal(
        const RealScalar& prec = Eigen::NumTraits<Scalar>::dummy_precision()
    ) const {
        return m_inner.isDiagonal(prec);
    }
    bool isIdentity(
        const RealScalar& prec = Eigen::NumTraits<Scalar>::dummy_precision()
    ) const {
        return m_inner.isIdentity(prec);
    }
    bool isLowerTriangular(
        const RealScalar& prec = Eigen::NumTraits<Scalar>::dummy_precision()
    ) const {
        return m_inner.isLowerTriangular(prec);
    }
    bool isUnitary(
        const RealScalar& prec = Eigen::NumTraits<Scalar>::dummy_precision()
    ) const {
        return m_inner.isUnitary(prec);
    }
    bool isUpperTriangular(
        const RealScalar& prec = Eigen::NumTraits<Scalar>::dummy_precision()
    ) const {
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
    Matrix& operator*=(const Eigen::MatrixBase<OtherDerived>& other) {
        m_inner *= other;
        return *this;
    }
    template <typename OtherScalar,
              int OtherRows,
              int OtherCols,
              int OtherOptions,
              int OtherMaxRows,
              int OtherMaxCols>
    Matrix& operator*=(const Matrix<OtherScalar,
                                    OtherRows,
                                    OtherCols,
                                    OtherOptions,
                                    OtherMaxRows,
                                    OtherMaxCols>& other) {
        // Aliasing can happen in in-place multiplication
        m_inner *= other.m_inner;
        return *this;
    }
    template <typename OtherDerived>
    Matrix& operator+=(const Eigen::EigenBase<OtherDerived>& other) {
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
    Matrix& operator-=(const Eigen::EigenBase<OtherDerived>& other) {
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

    // TODO: Extract PlainObjectBase to another class, we'll need that for
    //       Array support (and it'll be a good test drive for what follows)
    // TODO: Extract commonalities of DenseBase and below to another class,
    //       we'll need that for Array support
    // TODO: Replicate interface of Eigen::DenseBase
    // TODO: Replicate interface of all Eigen::DenseCoeffsBase types
    // TODO: Replicate interface of Eigen::EigenBase

private:
    Inner m_inner;

    static const RealScalar s_dummy_precision =
        Eigen::NumTraits<Scalar>::dummy_precision();
};

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
