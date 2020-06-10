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
#include "ForwardDeclarations.hpp"
#include "TypeTraits.hpp"
#include "CommaInitializer.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Spiritual equivalent of the superclasses of Eigen::Matrix
//
// Since we do not currently provide an Eigen::Array wrapper, this is slightly
// specialized towards the needs of Eigen::Matrix, and therefore less general
// than Eigen::PlainObjectBase (hence the different name). But it can be turned
// into a true Eigen::PlainObjectBase equivalent if needed.
//
template <typename Derived>
class PlainMatrixBase {
private:
    // Eigen type wrapped by the CRTP daughter class
    using DerivedTraits = TypeTraits<Derived>;
    using Inner = typename DerivedTraits::Inner;

    // Template parameters of derived class
    static constexpr int Rows = DerivedTraits::Rows;
    static constexpr int Cols = DerivedTraits::Cols;
    static constexpr int Options = DerivedTraits::Options;
    static constexpr int MaxRows = DerivedTraits::MaxRows;
    static constexpr int MaxCols = DerivedTraits::MaxCols;

    // Eigen convenience
    using RealScalar = typename Inner::RealScalar;

public:
    // Derived class scalar type
    using Scalar = typename DerivedTraits::Scalar;

private:
    // Quick way to construct a matrix of the same size, but w/o the options
    using PlainBase = Matrix<Scalar, Rows, Cols>;

public:
    // === Eigen::EigenBase interface ===

    // Index convenience typedef
    using Index = Eigen::Index;

    // Matrix dimensions
    Index cols() const {
        return derivedInner().cols();
    }
    Index rows() const {
        return derivedInner().rows();
    }
    Index size() const {
        return derivedInner().size();
    }

    // CRTP daughter class access
    Derived& derived() {
        return *static_cast<Derived*>(this);
    }
    const Derived& derived() const {
        return *static_cast<const Derived*>(this);
    }

    // === Eigen::DenseCoeffsBase interfaces ===

    // Storage layout (DirectWriteAccessors specific)
    Index colStride() const {
        return derivedInner().colStride();
    }
    Index innerStride() const {
        return derivedInner().innerStride();
    }
    Index outerStride() const {
        return derivedInner().outerStride();
    }
    Index rowStride() const {
        return derivedInner().rowStride();
    }

    // Access elements for writing (WriteAccessors specific)
    template <typename... Index>
    Scalar& coeffRef(Index... indices) {
        return derivedInner().coeffRef(indices...);
    }
    template <typename... Index>
    Scalar& operator()(Index... indices) {
        return derivedInner()(indices...);
    }
    Scalar& operator[](Index index) {
        return derivedInner()[index];
    }
    Scalar& w() {
        return derivedInner().w();
    }
    Scalar& x() {
        return derivedInner().x();
    }
    Scalar& y() {
        return derivedInner().y();
    }
    Scalar& z() {
        return derivedInner().z();
    }

    // Read elements (ReadOnlyAccessors specific)
    template <typename... Index>
    decltype(auto) coeff(Index... indices) const {
        return derivedInner().coeff(indices...);
    }
    template <typename... Index>
    decltype(auto) operator()(Index... indices) const {
        return derivedInner()(indices...);
    }
    decltype(auto) operator[](Index index) const {
        return derivedInner()[index];
    }
    decltype(auto) w() const {
        return derivedInner().w();
    }
    decltype(auto) x() const {
        return derivedInner().x();
    }
    decltype(auto) y() const {
        return derivedInner().y();
    }
    decltype(auto) z() const {
        return derivedInner().z();
    }

    // === Eigen::DenseBase interface ===

    // Eigen-style compilation constants
    static constexpr int RowsAtCompileTime = Rows;
    static constexpr int ColsAtCompileTime = Cols;
    static constexpr int SizeAtCompileTime = Inner::SizeAtCompileTime;
    static constexpr int MaxRowsAtCompileTime = MaxRows;
    static constexpr int MaxColsAtCompileTime = MaxCols;
    static constexpr int MaxSizeAtCompileTime = Inner::MaxSizeAtCompileTime;
    static constexpr bool IsVectorAtCompileTime = Inner::IsVectorAtCompileTime;
    static constexpr unsigned int Flags = Inner::Flags;
    static constexpr bool IsRowMajor = Inner::IsRowMajor;

private:
    static constexpr int PlainMatrixOptions =
        AutoAlign | (Flags & (IsRowMajor ? RowMajor : ColMajor));

public:
    // Eigen-style typedefs
    // NOTE: Scalar was defined above
    // TODO: Add PlainArray support if Array support is added
    using PlainMatrix = Matrix<Scalar,
                               Rows,
                               Cols,
                               PlainMatrixOptions,
                               MaxRows,
                               MaxCols>;
    // TODO: Adapt if PlainArray support is added
    using PlainObject = PlainMatrix;
    using StorageIndex = typename Inner::StorageIndex;
    using value_type = Scalar;

    // "Array of bool" interface
    bool all() const {
        return derivedInner().all();
    }
    bool any() const {
        return derivedInner().any();
    }
    Index count() const {
        return derivedInner().count();
    }
    template <typename ThenDerived, typename ElseDerived>
    PlainObject select(const DenseBase<ThenDerived>& thenMatrix,
                       const DenseBase<ElseDerived>& elseMatrix) const {
        return PlainObject(derivedInner().select(thenMatrix.derivedInner(),
                                                 elseMatrix.derivedInner()));
    }
    template <typename ThenDerived, typename ElseDerived>
    PlainObject select(
        const Eigen::DenseBase<ThenDerived>& thenMatrix,
        const DenseBase<ElseDerived>& elseMatrix
    ) const {
        return PlainObject(derivedInner().select(thenMatrix,
                                                 elseMatrix.derivedInner()));
    }
    template <typename ThenDerived, typename ElseDerived>
    PlainObject select(
        const DenseBase<ThenDerived>& thenMatrix,
        const Eigen::DenseBase<ElseDerived>& elseMatrix
    ) const {
        return PlainObject(derivedInner().select(thenMatrix.derivedInner(),
                                                 elseMatrix));
    }
    template <typename ThenDerived, typename ElseDerived>
    PlainObject select(
        const Eigen::DenseBase<ThenDerived>& thenMatrix,
        const Eigen::DenseBase<ElseDerived>& elseMatrix
    ) const {
        return PlainObject(derivedInner().select(thenMatrix, elseMatrix));
    }
    template <typename ThenDerived>
    PlainObject select(
        const DenseBase<ThenDerived>& thenMatrix,
        const typename ThenDerived::Scalar& elseScalar
    ) const {
        return PlainObject(derivedInner().select(thenMatrix.derivedInner(),
                                                 elseScalar));
    }
    template <typename ThenDerived>
    PlainObject select(
        const Eigen::DenseBase<ThenDerived>& thenMatrix,
        const typename ThenDerived::Scalar& elseScalar
    ) const {
        return PlainObject(derivedInner().select(thenMatrix, elseScalar));
    }
    template <typename ElseDerived>
    PlainObject select(
        const typename ElseDerived::Scalar& thenScalar,
        const DenseBase<ElseDerived>& elseMatrix
    ) const {
        return PlainObject(derivedInner().select(thenScalar,
                                                 elseMatrix.derivedInner()));
    }
    template <typename ElseDerived>
    PlainObject select(
        const typename ElseDerived::Scalar& thenScalar,
        const Eigen::DenseBase<ElseDerived>& elseMatrix
    ) const {
        return PlainObject(derivedInner().select(thenScalar, elseMatrix));
    }

    // IEEE 754 error handling stupidity
    bool allFinite() const {
        return derivedInner().allFinite();
    }
    bool hasNaN() const {
        return derivedInner().hasNaN();
    }

    // FIXME: Support colwise() and rowwise(), which are used by Acts

    // eval() is only provided for Eigen API compatibility reasons, it is
    // perfectly useless here since eager evaluation is the default already
    Derived eval() const {
        return derivedInner();
    }

    // Filling various patterns into the inner data
    void fill(const Scalar& value) {
        derivedInner().fill(value);
    }
    Derived& setConstant(const Scalar& value) {
        return derivedInner().setConstant(value);
    }
    Derived& setLinSpaced(const Scalar& low, const Scalar& high) {
        return derivedInner().setLinSpaced(low, high);
    }
    Derived& setLinSpaced(Index size, const Scalar& low, const Scalar& high) {
        return derivedInner().setLinSpaced(size, low, high);
    }
    Derived& setOnes() {
        return derivedInner().setOnes();
    }
    Derived& setRandom() {
        return derivedInner().setRandom();
    }
    Derived& setZero() {
        return derivedInner().setZero();
    }

    // NOTE: flagged() is deprecated and unused by Acts, thus unsupported

    // TODO: Support format() text formatting controls
    //       These are not currently used by Acts, so lower-priority.

    // Size queries
    Index innerSize() const {
        return derivedInner().innerSize();
    }
    Index outerSize() const {
        return derivedInner().outerSize();
    }

    // Approximate comparisons
    template <typename OtherDerived>
    bool isApprox(const DenseBase<OtherDerived>& other,
                  const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isApprox(other.derivedInner(), prec);
    }
    template <typename OtherDerived>
    bool isApprox(const Eigen::DenseBase<OtherDerived>& other,
                  const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isApprox(other, prec);
    }
    bool isApproxToConstant(const Scalar& value,
                            const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isApproxToConstant(value, prec);
    }
    bool isConstant(const Scalar& value,
                    const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isConstant(value, prec);
    }
    template <typename OtherDerived>
    bool isMuchSmallerThan(const DenseBase<OtherDerived>& other,
                           const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isMuchSmallerThan(other.derivedInner(), prec);
    }
    template <typename OtherDerived>
    bool isMuchSmallerThan(const Eigen::DenseBase<OtherDerived>& other,
                           const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isMuchSmallerThan(other, prec);
    }
    bool isMuchSmallerThan(const RealScalar& other,
                           const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isMuchSmallerThan(other, prec);
    }
    bool isOnes(const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isOnes(prec);
    }
    bool isZero(const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isZero(prec);
    }

    // Data reduction
    Scalar maxCoeff() const {
        return derivedInner().maxCoeff();
    }
    Scalar maxCoeff(Index* index) const {
        return derivedInner().maxCoeff(index);
    }
    Scalar maxCoeff(Index* row, Index* col) const {
        return derivedInner().maxCoeff(row, col);
    }
    Scalar mean() const {
        return derivedInner().mean();
    }
    Scalar minCoeff() const {
        return derivedInner().minCoeff();
    }
    Scalar minCoeff(Index* index) const {
        return derivedInner().minCoeff(index);
    }
    Scalar minCoeff(Index* row, Index* col) const {
        return derivedInner().minCoeff(row, col);
    }
    Index nonZeros() {
        return derivedInner().nonZeros();
    }
    Scalar prod() const {
        return derivedInner().prod();
    }
    template <typename Func>
    Scalar redux(const Func& func) const {
        return derivedInner().redux(func);
    }
    Scalar sum() const {
        return derivedInner().sum();
    }
    Scalar value() const {
        return derivedInner().value();
    }

    // TODO: Understand and support nestByValue()
    //       This is not currently used by Acts, so lower-priority.

    // Comma initializer
    template <typename OtherDerived>
    CommaInitializer<Derived> operator<<(const DenseBase<OtherDerived>& other) {
        return CommaInitializer(derived(), other);
    }
    template <typename OtherDerived>
    CommaInitializer<Derived>
    operator<<(const Eigen::DenseBase<OtherDerived>& other) {
        return CommaInitializer(derived(), other);
    }
    CommaInitializer<Derived> operator<<(const Scalar& s) {
        return CommaInitializer(derived(), s);
    }

    // Assignment operator
    template <typename OtherDerived = Derived>
    Derived& operator=(const DenseBase<OtherDerived>& other) {
        derivedInner() = other.derivedInner();
        return *this;
    }
    template <typename OtherDerived = Inner>
    Derived& operator=(const Eigen::DenseBase<OtherDerived>& other) {
        derivedInner() = other;
        return *this;
    }
    // NOTE: No need for an Eagen::EigenBase overload yet as we don't replicate
    //       that layer of the Eigen class hierarchy yet.
    template <typename OtherDerived>
    Derived& operator=(const Eigen::EigenBase<OtherDerived>& other) {
        derivedInner() = other;
        return *this;
    }

    // TODO: Support replicate()
    //       This is not currently used by Acts, so lower-priority.

    // Inner array resizing
    void resize(Index newSize) {
        derivedInner().resize(newSize);
    }
    void resize(Index rows, Index cols) {
        derivedInner().resize(rows, cols);
    }

    // TODO: Support reverse(InPlace)?()
    //       This is not currently used by Acts, so lower-priority.

    // Swapping matrices
    template <typename OtherDerived>
    void swap(const DenseBase<OtherDerived>& other) {
        return derivedInner().swap(other.derivedInner());
    }
    template <typename OtherDerived>
    void swap(const Eigen::DenseBase<OtherDerived>& other) {
        return derivedInner().swap(other);
    }
    template <typename OtherDerived>
    void swap(PlainMatrixBase<OtherDerived>& other) {
        return derivedInner().swap(other.derivedInner());
    }
    template <typename OtherDerived>
    void swap(Eigen::PlainObjectBase<OtherDerived>& other) {
        return derivedInner().swap(other);
    }

    // Matrix transposition
    // NOTE: Will need adaptations if array support is needed
    Matrix<Scalar, Cols, Rows> transpose() {
        return Matrix<Scalar, Cols, Rows>(derivedInner().transpose());
    }
    Matrix<Scalar, Cols, Rows> transpose() const {
        return Matrix<Scalar, Cols, Rows>(derivedInner().transpose());
    }
    void transposeInPlace() {
        derivedInner().transposeInPlace();
    }

    // Visit coefficients
    template <typename Visitor>
    void visit(Visitor& func) const {
        derivedInner().visit(func);
    }

    // TODO: Support unaryExpr
    //       This is not used by Acts, and therefore not a priority.

    // Special plain object generators
    template <typename... Args>
    static PlainObject Constant(Args&&... args) {
        return PlainObject(Inner::Constant(std::forward<Args>(args)...));
    }
    template <typename... Args>
    static PlainObject LinSpaced(Args&&... args) {
        return PlainObject(Inner::LinSpaced(std::forward<Args>(args)...));
    }
    template <typename... Args>
    static PlainObject NullaryExpr(Args&&... args) {
        return PlainObject(Inner::NullaryExpr(std::forward<Args>(args)...));
    }
    template <typename... Index>
    static PlainObject Ones(Index... indices) {
        return PlainObject(Inner::Ones(indices...));
    }
    template <typename... Index>
    static PlainObject Random(Index... indices) {
        return PlainObject(Inner::Random(indices...));
    }
    template <typename... Index>
    static PlainObject Zero(Index... indices) {
        return PlainObject(Inner::Zero(indices...));
    }

    // === Eigen::MatrixBase interface ===

    // Adjoint = conjugate + transpose
    Matrix<Scalar, Cols, Rows> adjoint() const {
        return Matrix<Scalar, Cols, Rows>(derivedInner().adjoint());
    }
    void adjointInPlace() {
        derivedInner().adjointInPlace();
    }

    // TODO: Support methods involving Householder transformations
    //       These are not currently used by Acts, so lower-priority.

    // In-place products
    template <typename... Args>
    void applyOnTheLeft(Args&&... args) {
        derivedInner().applyOnTheLeft(std::forward<Args>(args)...);
    }
    template <typename OtherDerived>
    void applyOnTheLeft(const EigenBase<OtherDerived>& other) {
        derivedInner().applyOnTheLeft(other.derivedInner());
    }
    template <typename... Args>
    void applyOnTheRight(Args&&... args) {
        derivedInner().applyOnTheRight(std::forward<Args>(args)...);
    }
    template <typename OtherDerived>
    void applyOnTheRight(const EigenBase<OtherDerived>& other) {
        derivedInner().applyOnTheRight(other.derivedInner());
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
        return derivedInner().array();
    }
    Eigen::ArrayWrapper<const Inner> array() const {
        return derivedInner().array();
    }

    // Reinterpret a vector as a diagonal matrix
    // TODO: Consider providing a dedicated diagonal wrapper type to avoid
    //       copies and extra run-time storage and computation overhead
    Matrix<Scalar, std::max(Rows, Cols), std::max(Rows, Cols)>
    asDiagonal() const {
        assert(std::min(Rows, Cols) == 1);
        return Matrix<Scalar, std::max(Rows, Cols), std::max(Rows, Cols)>(
            derivedInner().asDiagonal()
        );
    }
    // TODO: Support SVD-related methods bdcSvd() and jacobiSvd()
    //       These are not currently used by Acts, so lower-priority.

    // Norms and normalization
    RealScalar blueNorm() const {
        return derivedInner().blueNorm();
    }
    // TODO: Support hnormalized()
    //       This is not currently used by Acts, so lower-priority.
    RealScalar hypotNorm() const {
        return derivedInner().hypotNorm();
    }
    RealScalar lpNorm() const {
        return derivedInner().lpNorm();
    }
    RealScalar norm() const {
        return derivedInner().norm();
    }
    void normalize() {
        derivedInner().normalize();
    }
    PlainBase normalized() {
        return derivedInner().normalized();
    }
    RealScalar operatorNorm() const {
        return derivedInner().operatorNorm();
    }
    RealScalar squaredNorm() const {
        return derivedInner().squaredNorm();
    }
    RealScalar stableNorm() const {
        return derivedInner().stableNorm();
    }
    void stableNormalize() {
        derivedInner().stableNormalize();
    }
    PlainBase stableNormalized() const {
        return derivedInner().stableNormalized();
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
        return PlainBase(derivedInner().cross(other));
    }
    template <typename OtherDerived>
    PlainBase
    cross(const MatrixBase<OtherDerived>& other) const {
        return PlainBase(derivedInner().cross(other.derivedInner()));
    }
    template <typename OtherDerived>
    PlainBase
    cross3(const Eigen::MatrixBase<OtherDerived>& other) const {
        return PlainBase(derivedInner().cross3(other));
    }
    template <typename OtherDerived>
    PlainBase
    cross3(const MatrixBase<OtherDerived>& other) const {
        return PlainBase(derivedInner().cross3(other.derivedInner()));
    }

    // Determinant
    Scalar determinant() const {
        return derivedInner().determinant();
    }

    // Diagonal and associated properties
    // FIXME: Support non-const diagonal access, which is used by Acts for
    //        comma initialization based diagonal assignment.
    //        Use that to make the diagonal access method below more efficient.
    template <typename... Index>
    Vector<Scalar, Rows> diagonal(Index... indices) const {
        return Vector<Scalar, Rows>(derivedInner().diagonal(indices...));
    }
    Index diagonalSize() const {
        return derivedInner().diagonalSize();
    }

    // Dot product
    template <typename OtherDerived>
    Scalar dot(const Eigen::MatrixBase<OtherDerived>& other) const {
        return derivedInner().dot(other);
    }
    template <typename OtherDerived>
    Scalar dot(const MatrixBase<OtherDerived>& other) const {
        return derivedInner().dot(other.derivedInner());
    }

    // Eigenvalues
    Vector<Scalar, Rows> eigenvalues() const {
        return Vector<Scalar, Rows>(derivedInner().eigenvalues());
    }

    // Euler angles
    Vector<Scalar, 3> eulerAngles(Index a0, Index a1, Index a2) const {
        return Vector<Scalar, 3>(derivedInner().eulerAngles());
    }

    // TODO: Support aligned access enforcement
    //       This is not currently used by Acts, so lower-priority.
    // TODO: Support LU decomposition
    //       This is not currently used by Acts, so lower-priority.
    // TODO: Support homogeneous()
    //       This is not currently used by Acts, so lower-priority.

    // Matrix inversion
    PlainBase inverse() const {
        return PlainBase(derivedInner().inverse());
    }

    // Approximate comparisons
    bool isDiagonal(const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isDiagonal(prec);
    }
    bool isIdentity(const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isIdentity(prec);
    }
    bool isLowerTriangular(const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isLowerTriangular(prec);
    }
    bool isUnitary(const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isUnitary(prec);
    }
    bool isUpperTriangular(const RealScalar& prec = dummy_precision()) const {
        return derivedInner().isUpperTriangular(prec);
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
        return (derivedInner() == other);
    }
    template <typename OtherDerived>
    bool operator==(const MatrixBase<OtherDerived>& other) const {
        return (derivedInner() == other.derivedInner());
    }
    template <typename Other>
    bool operator!=(const Other& other) const {
        return !(*this == other);
    }

    // Matrix-scalar multiplication
    template <typename OtherScalar>
    PlainBase operator*(const OtherScalar& scalar) const {
        return PlainBase(derivedInner() * scalar);
    }
    template <typename OtherScalar>
    friend PlainBase operator*(const OtherScalar& scalar, const Derived& matrix) {
        return matrix * scalar;
    }
    template <typename OtherScalar>
    Derived& operator*=(const OtherScalar& scalar) {
        derivedInner() *= scalar;
        return *this;
    }

    // Matrix-scalar division
    template <typename OtherScalar>
    PlainBase operator/(const OtherScalar& scalar) const {
        return PlainBase(derivedInner() / scalar);
    }
    template <typename OtherScalar>
    Derived& operator/=(const OtherScalar& scalar) {
        derivedInner() /= scalar;
        return *this;
    }

    // Matrix-matrix multiplication
    template <typename DiagonalDerived>
    PlainBase
    operator*(const Eigen::DiagonalBase<DiagonalDerived>& other) const {
        return PlainBase(derivedInner() * other);
    }
    template <typename OtherDerived>
    Matrix<Scalar, Rows, OtherDerived::ColsAtCompileTime>
    operator*(const Eigen::MatrixBase<OtherDerived>& other) const {
        return Matrix<Scalar, Rows, OtherDerived::ColsAtCompileTime>(
            derivedInner() * other
        );
    }
    template <typename OtherDerived>
    Matrix<Scalar, Rows, OtherDerived::Cols>
    operator*(const MatrixBase<OtherDerived>& other) const {
        return Matrix<Scalar, Rows, OtherDerived::Cols>(
            derivedInner() * other.derivedInner()
        );
    }
    template <typename OtherDerived>
    Derived& operator*=(const Eigen::EigenBase<OtherDerived>& other) {
        derivedInner() *= other;
        return *this;
    }
    template <typename OtherDerived>
    Derived& operator*=(const EigenBase<OtherDerived>& other) {
        derivedInner() *= other.derivedInner();
        return *this;
    }

    // Matrix addition
    template <typename OtherEigen>
    PlainBase operator+(const OtherEigen& other) const {
        return PlainBase(derivedInner() + other);
    }
    template <typename OtherDerived>
    PlainBase operator+(const MatrixBase<OtherDerived>& other) const {
        return PlainBase(derivedInner() + other.derivedInner());
    }
    template <typename OtherDerived>
    Derived& operator+=(const Eigen::MatrixBase<OtherDerived>& other) {
        derivedInner() += other;
        return *this;
    }
    template <typename OtherDerived>
    Derived& operator+=(const MatrixBase<OtherDerived>& other) {
        derivedInner() += other.derivedInner();
        return *this;
    }

    // Matrix subtraction
    template <typename OtherEigen>
    PlainBase operator-(const OtherEigen& other) const {
        return PlainBase(derivedInner() + other);
    }
    template <typename OtherDerived>
    PlainBase operator-(const MatrixBase<OtherDerived>& other) const {
        return PlainBase(derivedInner() + other.derivedInner());
    }
    template <typename OtherDerived>
    Derived& operator-=(const Eigen::MatrixBase<OtherDerived>& other) {
        derivedInner() -= other;
        return *this;
    }
    template <typename OtherDerived>
    Derived& operator-=(const MatrixBase<OtherDerived>& other) {
        derivedInner() -= other.derivedInner();
        return *this;
    }

    // TODO: Support selfadjointView()
    //       This is not currently used by Acts, so lower-priority.

    // Set to identity matrix
    template <typename... Index>
    Derived& setIdentity(Index... indices) {
        derivedInner().setIdentity(indices...);
        return *this;
    }

    // TODO: Support sparse matrices
    //       This is not currently used by Acts, so lower-priority.

    // Matrix trace
    Scalar trace() const {
        return derivedInner().trace();
    }

    // TODO: Support triangular view
    //       This is not currently used by Acts, so lower-priority.

    // Coefficient-wise operations
    template <typename Other>
    PlainBase cwiseMin(const Other& other) const {
        return PlainBase(derivedInner().cwiseMin(other));
    }
    template <typename OtherDerived>
    PlainBase cwiseMin(const MatrixBase<OtherDerived>& other) const {
        return PlainBase(derivedInner().cwiseMin(other.derivedInner()));
    }
    template <typename Other>
    PlainBase cwiseMax(const Other& other) const {
        return PlainBase(derivedInner().cwiseMax(other));
    }
    template <typename OtherDerived>
    PlainBase cwiseMax(const MatrixBase<OtherDerived>& other) const {
        return PlainBase(derivedInner().cwiseMax(other.derivedInner()));
    }
    Matrix<RealScalar, Rows, Cols> cwiseAbs2() const {
        return Matrix<RealScalar, Rows, Cols>(derivedInner().cwiseAbs2());
    }
    Matrix<RealScalar, Rows, Cols> cwiseAbs() const {
        return Matrix<RealScalar, Rows, Cols>(derivedInner().cwiseAbs());
    }
    PlainBase cwiseSqrt() const {
        return PlainBase(derivedInner().cwiseSqrt());
    }
    PlainBase cwiseInverse() const {
        return PlainBase(derivedInner().cwiseInverse());
    }
    template <typename OtherDerived>
    PlainBase cwiseProduct(const Eigen::MatrixBase<OtherDerived>& other) const {
        return PlainBase(derivedInner().cwiseProduct(other));
    }
    template <typename OtherDerived>
    PlainBase cwiseProduct(const MatrixBase<OtherDerived>& other) const {
        return PlainBase(derivedInner().cwiseProduct(other.derivedInner()));
    }
    template <typename OtherDerived>
    PlainBase cwiseQuotient(const Eigen::MatrixBase<OtherDerived>& other) const {
        return PlainBase(derivedInner().cwiseQuotient(other));
    }
    template <typename OtherDerived>
    PlainBase cwiseQuotient(const MatrixBase<OtherDerived>& other) const {
        return PlainBase(derivedInner().cwiseQuotient(other.derivedInner()));
    }
    template <typename Other>
    PlainBase cwiseEqual(const Other& other) const {
        return PlainBase(derivedInner().cwiseEqual(other));
    }
    template <typename OtherDerived>
    PlainBase cwiseEqual(const MatrixBase<OtherDerived>& other) const {
        return PlainBase(derivedInner().cwiseEqual(other.derivedInner()));
    }
    template <typename OtherDerived>
    PlainBase cwiseNotEqual(const Eigen::MatrixBase<OtherDerived>& other) const {
        return PlainBase(derivedInner().cwiseNotEqual(other));
    }
    template <typename OtherDerived>
    PlainBase cwiseNotEqual(const MatrixBase<OtherDerived>& other) const {
        return PlainBase(derivedInner().cwiseNotEqual(other.derivedInner()));
    }

    // Special matrix generators
    template <typename... Index>
    static Derived Identity(Index... indices) {
        return Derived(Inner::Identity(indices...));
    }
    template <typename... Index>
    static Derived Unit(Index... indices) {
        return Derived(Inner::Unit(indices...));
    }
    static Derived UnitW() {
        return Derived(Inner::UnitW());
    }
    static Derived UnitX() {
        return Derived(Inner::UnitX());
    }
    static Derived UnitY() {
        return Derived(Inner::UnitY());
    }
    static Derived UnitZ() {
        return Derived(Inner::UnitZ());
    }

    // === Eigen::PlainObjectBase interface ===

    // Resizing
    template <typename... Args>
    void conservativeResize(Args... args) {
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
    template <typename... Args>
    void resize(Args... args) {
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
    template <typename... Args>
    Derived& setConstant(Args&&... args) {
        derivedInner().setConstant(std::forward<Args>(args)...);
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
    template <typename... Args>
    static Derived Map(Args&&... args) {
        return Derived(Inner::Map(std::forward<Args>(args)...));
    }
    template <typename... Args>
    static Derived MapAligned(Args&&... args) {
        return Derived(Inner::MapAligned(std::forward<Args>(args)...));
    }

    // === Coefficient-wise ops, unknown Eigen base class ===
    //
    // FIXME: I'm not sure which base class these methods should belong to, but
    //        Eigen docs claim that all expressions should have it, so perhaps
    //        EigenBase would be the right pick?
    //
    PlainObject real() const {
        return PlainObject(derivedInner().real());
    }
    PlainObject imag() const {
        return PlainObject(derivedInner().imag());
    }
    PlainObject conjugate() const {
        return PlainObject(derivedInner().conjugate());
    }

protected:
    // Access the inner Eigen object held by the CRTP daughter class
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
    static RealScalar dummy_precision() {
        return Eigen::NumTraits<Scalar>::dummy_precision();
    }
};

// Display of DenseBase and above
template <typename Derived>
std::ostream& operator<<(std::ostream& s, const DenseBase<Derived>& m) {
    return s << m.derived().getInner();
}

}  // namespace Eagen

}  // namespace detail

}  // namespace Acts
