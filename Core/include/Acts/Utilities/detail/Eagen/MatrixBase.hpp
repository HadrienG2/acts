// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "DiagonalBase.hpp"
#include "DiagonalMatrix.hpp"
#include "EigenBase.hpp"
#include "EigenDense.hpp"
#include "EigenPrologue.hpp"
#include "ForwardDeclarations.hpp"
#include "TypeTraits.hpp"
#include "CommaInitializer.hpp"

namespace Acts {

namespace detail {

namespace Eagen {

// Spiritual equivalent of Eigen::MatrixBase
//
// This is slightly specialized towards the needs of Eagen::Matrix, in
// particular it always exposes the full direct write DenseCoeffsBase interface.
// If the need arises, that can be changed.
//
template <typename _Derived>
class MatrixBase : public EigenBase<_Derived> {
    using Super = EigenBase<_Derived>;

public:
    // === Eagen wrapper API ===

    // Re-expose template parameters
    using Derived = _Derived;

    // Wrapped Eigen type
    using Inner = typename Super::Inner;

    // Re-expose Matrix typedefs and constexprs
protected:
    using DerivedTraits = typename Super::DerivedTraits;
public:
    using Scalar = typename DerivedTraits::Scalar;
    static constexpr int Rows = DerivedTraits::Rows;
    static constexpr int Cols = DerivedTraits::Cols;
    static constexpr int MaxRows = DerivedTraits::MaxRows;
    static constexpr int MaxCols = DerivedTraits::MaxCols;

    // === Base class API ===

    // Re-export useful base class interface
    using Index = typename Super::Index;
    using Super::cols;
    using Super::derived;
    using Super::derivedInner;
    using Super::rows;
    using Super::size;

    // === Eigen::DenseCoeffsBase interfaces ===

    // Real version of the Scalar type
private:
    using ScalarTraits = NumTraits<Scalar>;
public:
    using RealScalar = typename ScalarTraits::Real;

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
    //
    // NOTE: To simplify the implementation, Eagen does not strictly replicate
    //       the DenseCoeffsBase base class stack, but instead lets Eigen do the
    //       hard work of figuring out the right return type for each operation.
    //       So e.g. for Map<const T>, these "non-const" accessors will actually
    //       call the Eigen const accessor and return a const value/reference.
    //
    template <typename... Index>
    decltype(auto) coeffRef(Index... indices) {
        return derivedInner().coeffRef(indices...);
    }
    template <typename... Index>
    decltype(auto) operator()(Index... indices) {
        return derivedInner()(indices...);
    }
    decltype(auto) operator[](Index index) {
        return derivedInner()[index];
    }
    decltype(auto) w() {
        return derivedInner().w();
    }
    decltype(auto) x() {
        return derivedInner().x();
    }
    decltype(auto) y() {
        return derivedInner().y();
    }
    decltype(auto) z() {
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

    // Eigen-style typedefs
    // NOTE: Scalar was defined above
private:
    static constexpr int PlainMatrixOptions =
        AutoAlign | (Flags & (IsRowMajor ? RowMajor : ColMajor));
public:
    using PlainMatrix = Matrix<Scalar,
                               Rows,
                               Cols,
                               PlainMatrixOptions,
                               MaxRows,
                               MaxCols>;
    // TODO: Add PlainArray support and adjust PlainObject typedef if Array
    //       support is added to Eagen.
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
    // TODO: Roll out a simpler select() if Array support is added to Eagen.
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

    // Materialize this expression into its own plain object
    PlainObject eval() const {
        return PlainObject(derivedInner());
    }

    // Filling various patterns into the inner data
    void fill(const Scalar& value) {
        derivedInner().fill(value);
    }
    Derived& setConstant(const Scalar& value) {
        derivedInner().setConstant(value);
        return derived();
    }
    Derived& setLinSpaced(const Scalar& low, const Scalar& high) {
        derivedInner().setLinSpaced(low, high);
        return derived();
    }
    Derived& setLinSpaced(Index size, const Scalar& low, const Scalar& high) {
        derivedInner().setLinSpaced(size, low, high);
        return derived();
    }
    Derived& setOnes() {
        derivedInner().setOnes();
        return derived();
    }
    Derived& setRandom() {
        derivedInner().setRandom();
        return derived();
    }
    Derived& setZero() {
        derivedInner().setZero();
        return derived();
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
    template<typename IndexType>
    Scalar maxCoeff(IndexType* index) const {
        return derivedInner().maxCoeff(index);
    }
    template<typename IndexType>
    Scalar maxCoeff(IndexType* row, IndexType* col) const {
        return derivedInner().maxCoeff(row, col);
    }
    Scalar mean() const {
        return derivedInner().mean();
    }
    Scalar minCoeff() const {
        return derivedInner().minCoeff();
    }
    template<typename IndexType>
    Scalar minCoeff(IndexType* index) const {
        return derivedInner().minCoeff(index);
    }
    template<typename IndexType>
    Scalar minCoeff(IndexType* row, IndexType* col) const {
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
    Derived& operator=(const EigenBase<OtherDerived>& other) {
        derivedInner() = other.derivedInner();
        return derived();
    }
    template <typename OtherDerived = Inner>
    Derived& operator=(const Eigen::EigenBase<OtherDerived>& other) {
        derivedInner() = other;
        return derived();
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
    template <typename OtherDerived>
    void applyOnTheLeft(const EigenBase<OtherDerived>& other) {
        derivedInner().applyOnTheLeft(other.derivedInner());
    }
    template <typename OtherDerived>
    void applyOnTheLeft(const Eigen::EigenBase<OtherDerived>& other) {
        derivedInner().applyOnTheLeft(other);
    }
    // FIXME: No Eagen equivalent here
    template <typename OtherScalar>
    void applyOnTheLeft(Index p,
                        Index q,
                        const Eigen::JacobiRotation<OtherScalar>& j) {
        derivedInner().applyOnTheLeft(p, q, j);
    }
    template <typename OtherDerived>
    void applyOnTheRight(const EigenBase<OtherDerived>& other) {
        derivedInner().applyOnTheRight(other.derivedInner());
    }
    template <typename OtherDerived>
    void applyOnTheRight(const Eigen::EigenBase<OtherDerived>& other) {
        derivedInner().applyOnTheRight(other);
    }
    // FIXME: No Eagen equivalent here
    template <typename OtherScalar>
    void applyOnTheRight(Index p,
                         Index q,
                         const Eigen::JacobiRotation<OtherScalar>& j) {
        derivedInner().applyOnTheRight(p, q, j);
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
    DiagonalWrapper<const Derived>
    asDiagonal() const {
        assert(std::min(Rows, Cols) == 1);
        return DiagonalWrapper<const Derived>(derived());
    }

    // Perform SVD decomposition
    JacobiSVD<PlainMatrix>
    jacobiSvd(unsigned int computationOptions = 0) const {
        return JacobiSVD<PlainMatrix>(derivedInner(), computationOptions);
    }
    // TODO: Support bdcSvd() style SVD decomposition
    //       This is not currently used by Acts, so lower-priority.

    // Norms and normalization
    RealScalar blueNorm() const {
        return derivedInner().blueNorm();
    }
    // TODO: Support hnormalized()
    //       This is not currently used by Acts, so lower-priority.
    RealScalar hypotNorm() const {
        return derivedInner().hypotNorm();
    }
    template <int p>
    RealScalar lpNorm() const {
        return derivedInner().template lpNorm<1>();
    }
    RealScalar norm() const {
        return derivedInner().norm();
    }
    void normalize() {
        derivedInner().normalize();
    }
    PlainMatrix normalized() const {
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
    PlainMatrix stableNormalized() const {
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
    PlainMatrix
    cross(const MatrixBase<OtherDerived>& other) const {
        return PlainMatrix(derivedInner().cross(other.derivedInner()));
    }
    template <typename OtherDerived>
    PlainMatrix
    cross3(const MatrixBase<OtherDerived>& other) const {
        return PlainMatrix(derivedInner().cross3(other.derivedInner()));
    }

    // Determinant
    Scalar determinant() const {
        return derivedInner().determinant();
    }

    // Diagonal and associated properties
    using DiagonalReturnType = Diagonal<Derived>;
    template <int DiagIndex = 0>
    Diagonal<Derived, DiagIndex> diagonal() {
        return Diagonal(derived());
    }
    template <int DiagIndex = 0>
    Diagonal<const Derived, DiagIndex> diagonal() const {
        return Diagonal(derived());
    }

    // Size of the matrix' diagonal
    Index diagonalSize() const {
        return derivedInner().diagonalSize();
    }

    // Dot product
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
        return Vector<Scalar, 3>(derivedInner().eulerAngles(a0, a1, a2));
    }

    // TODO: Support aligned access enforcement
    //       This is not currently used by Acts, so lower-priority.
    // TODO: Support LU decomposition
    //       This is not currently used by Acts, so lower-priority.
    // TODO: Support homogeneous()
    //       This is not currently used by Acts, so lower-priority.

    // Matrix inversion
    PlainMatrix inverse() const {
        return PlainMatrix(derivedInner().inverse());
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

    // Cholesky decompositions
    LDLT<PlainMatrix> ldlt() const {
        return LDLT(derivedInner());
    }
    LLT<PlainMatrix> llt() const {
        return LLT(derivedInner());
    }

    // NOTE: noalias() will probably never be supported, as it should almost
    //       never be needed in an eagerly evaluated world, where AFAIK its only
    //       use is to make operator*= slightly more efficient.

    // Equality and inequality
    template <typename OtherDerived>
    bool operator==(const MatrixBase<OtherDerived>& other) const {
        return (derivedInner() == other.derivedInner());
    }
    template <typename Other>
    bool operator!=(const Other& other) const {
        return !(*this == other);
    }

    // Matrix-scalar multiplication
    PlainMatrix operator*(const Scalar& scalar) const {
        return PlainMatrix(derivedInner() * scalar);
    }
    friend PlainMatrix operator*(const Scalar& scalar, const Derived& matrix) {
        return matrix * scalar;
    }
    Derived& operator*=(const Scalar& scalar) {
        derivedInner() *= scalar;
        return derived();
    }

    // Matrix-scalar division
    PlainMatrix operator/(const Scalar& scalar) const {
        return PlainMatrix(derivedInner() / scalar);
    }
    Derived& operator/=(const Scalar& scalar) {
        derivedInner() /= scalar;
        return derived();
    }

    // Edge case of handling a 1x1 matrix as a scalar
    template <typename OtherDerived>
    PlainMatrix operator/(const MatrixBase<OtherDerived>& other) const {
        return PlainMatrix(derivedInner() / other.derivedInner().value());
    }
    template <typename OtherDerived>
    Derived& operator/=(const Eigen::EigenBase<OtherDerived>& other) {
        derivedInner() /= other.value();
        return derived();
    }
    template <typename OtherDerived>
    Derived& operator/=(const EigenBase<OtherDerived>& other) {
        derivedInner() /= other.derivedInner().value();
        return derived();
    }

    // Matrix-matrix multiplication
    template <typename DiagonalDerived>
    PlainMatrix
    operator*(const DiagonalBase<DiagonalDerived>& other) const {
        return PlainMatrix(derivedInner() * other.derivedInner());
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
        return derived();
    }
    template <typename OtherDerived>
    Derived& operator*=(const EigenBase<OtherDerived>& other) {
        derivedInner() *= other.derivedInner();
        return derived();
    }

    // Matrix addition
    template <typename OtherDerived>
    PlainMatrix operator+(const MatrixBase<OtherDerived>& other) const {
        return PlainMatrix(derivedInner() + other.derivedInner());
    }
    template <typename OtherDerived>
    Derived& operator+=(const MatrixBase<OtherDerived>& other) {
        derivedInner() += other.derivedInner();
        return derived();
    }

    // Edge case of summing a 1x1 matrix with a scalar
    Scalar operator+(const Scalar& s) const {
        return derivedInner().value() + s;
    }
    Derived& operator+=(const Scalar& s) {
        derivedInner() += Eigen::Matrix<Scalar, 1, 1>(s);
        return derived();
    }

    // Matrix subtraction
    template <typename OtherDerived>
    PlainMatrix operator-(const MatrixBase<OtherDerived>& other) const {
        return PlainMatrix(derivedInner() - other.derivedInner());
    }
    template <typename OtherDerived>
    Derived& operator-=(const MatrixBase<OtherDerived>& other) {
        derivedInner() -= other.derivedInner();
        return derived();
    }

    // Matrix negation
    PlainMatrix operator-() const {
        return PlainMatrix(-derivedInner());
    }

    // TODO: Support selfadjointView()
    //       This is not currently used by Acts, so lower-priority.

    // Set to identity matrix
    template <typename... Index>
    Derived& setIdentity(Index... indices) {
        derivedInner().setIdentity(indices...);
        return derived();
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
    template <typename OtherDerived>
    PlainMatrix cwiseMin(const MatrixBase<OtherDerived>& other) const {
        return PlainMatrix(derivedInner().cwiseMin(other.derivedInner()));
    }
    template <typename OtherDerived>
    PlainMatrix cwiseMax(const MatrixBase<OtherDerived>& other) const {
        return PlainMatrix(derivedInner().cwiseMax(other.derivedInner()));
    }
    Matrix<RealScalar, Rows, Cols> cwiseAbs2() const {
        return Matrix<RealScalar, Rows, Cols>(derivedInner().cwiseAbs2());
    }
    Matrix<RealScalar, Rows, Cols> cwiseAbs() const {
        return Matrix<RealScalar, Rows, Cols>(derivedInner().cwiseAbs());
    }
    PlainMatrix cwiseSqrt() const {
        return PlainMatrix(derivedInner().cwiseSqrt());
    }
    PlainMatrix cwiseInverse() const {
        return PlainMatrix(derivedInner().cwiseInverse());
    }
    template <typename OtherDerived>
    PlainMatrix cwiseProduct(const MatrixBase<OtherDerived>& other) const {
        return PlainMatrix(derivedInner().cwiseProduct(other.derivedInner()));
    }
    template <typename OtherDerived>
    PlainMatrix cwiseQuotient(const MatrixBase<OtherDerived>& other) const {
        return PlainMatrix(derivedInner().cwiseQuotient(other.derivedInner()));
    }
    template <typename OtherDerived>
    PlainMatrix cwiseEqual(const MatrixBase<OtherDerived>& other) const {
        return PlainMatrix(derivedInner().cwiseEqual(other.derivedInner()));
    }
    template <typename OtherDerived>
    PlainMatrix cwiseNotEqual(const MatrixBase<OtherDerived>& other) const {
        return PlainMatrix(derivedInner().cwiseNotEqual(other.derivedInner()));
    }

    // Special matrix generators
    template <typename... Index>
    static PlainMatrix Identity(Index... indices) {
        return PlainMatrix(Inner::Identity(indices...));
    }
    template <typename... Index>
    static PlainMatrix Unit(Index... indices) {
        return PlainMatrix(Inner::Unit(indices...));
    }
    static PlainMatrix UnitW() {
        return PlainMatrix(Inner::UnitW());
    }
    static PlainMatrix UnitX() {
        return PlainMatrix(Inner::UnitX());
    }
    static PlainMatrix UnitY() {
        return PlainMatrix(Inner::UnitY());
    }
    static PlainMatrix UnitZ() {
        return PlainMatrix(Inner::UnitZ());
    }

    // --- Sub-matrices ---

    // Sub-vector accessors
    template <int Length>
    using SubVectorBlock = VectorBlock<Derived, Length>;
    template <int Length>
    using ConstSubVectorBlock = VectorBlock<const Derived, Length>;
    SubVectorBlock<Dynamic> head(int n) {
        return segment(0, n);
    }
    ConstSubVectorBlock<Dynamic> head(int n) const {
        return segment(0, n);
    }
    template <int n>
    SubVectorBlock<n> head() {
        return segment<n>(0);
    }
    template <int n>
    ConstSubVectorBlock<n> head() const {
        return segment<n>(0);
    }
    SubVectorBlock<Dynamic> tail(int n) {
        return segment(size()-n, n);
    }
    ConstSubVectorBlock<Dynamic> tail(int n) const {
        return segment(size()-n, n);
    }
    template <int n>
    SubVectorBlock<n> tail() {
        return segment<n>(size()-n);
    }
    template <int n>
    ConstSubVectorBlock<n> tail() const {
        return segment<n>(size()-n);
    }
    SubVectorBlock<Dynamic> segment(Index pos, int n) {
        return SubVectorBlock<Dynamic>(derived(), pos, n);
    }
    ConstSubVectorBlock<Dynamic> segment(Index pos, int n) const {
        return ConstSubVectorBlock<Dynamic>(derived(), pos, n);
    }
    template <int n>
    SubVectorBlock<n> segment(Index pos) {
        return SubVectorBlock<n>(derived(), pos);
    }
    template <int n>
    ConstSubVectorBlock<n> segment(Index pos) const {
        return ConstSubVectorBlock<n>(derived(), pos);
    }

    // Sub-vector extractors (/!\ EAGEN EXTENSION /!\)
    //
    // Same idea as the templated sub-vector accessors above, but return an
    // owned sub-vector instead of a sub-vector view.
    //
    // This is not appropriate in all cases, as you must be careful not to build
    // dangling pointers to these owned vectors, and of course you should not
    // try to write to them. But when it works, it should be good for compile
    // time, without a noticeable run-time impact since compilers are actually
    // quite good at eliding copies.
    //
    // I'm more hesitant to do the same for dynamic-sized blocks, as
    // dynamic-sized vectors require memory allocation and compilers are less
    // good at removing that (clang manages sometimes, GCC almost never).
    //
    template <int Length>
    using SubVector = Vector<Scalar, Length>;
    template <int n>
    SubVector<n> extractHead() const {
        return extractSegment<n>(0);
    }
    template <int n>
    SubVector<n> extractTail() const {
        return extractSegment<n>(size() - n);
    }
    template <int n>
    SubVector<n> extractSegment(Index pos) const {
        SubVector<n> result;
        for (Index i = 0; i < n; ++i) {
            result.coeffRef(i) = coeff(i + pos);
        }
        return result;
    }

    // Sub-matrix accessors
private:
    template <typename DerivedType, int BlockRows, int BlockCols>
    using GenericSubMatrixBlock = Block<
        DerivedType,
        BlockRows,
        BlockCols,
        IsRowMajor ? ((BlockRows == 1)
                      || ((BlockCols != Dynamic) && (BlockCols == Cols)))
                   : ((BlockCols == 1)
                      || ((BlockRows != Dynamic) && (BlockRows == Rows)))
    >;
    template <int BlockRows, int BlockCols>
    using SubMatrixBlock = GenericSubMatrixBlock<Derived, BlockRows, BlockCols>;
    template <int BlockRows, int BlockCols>
    using ConstSubMatrixBlock = GenericSubMatrixBlock<const Derived,
                                                      BlockRows,
                                                      BlockCols>;
public:
    SubMatrixBlock<Dynamic, Dynamic> block(Index startRow,
                                           Index startCol,
                                           int blockRows,
                                           int blockCols) {
        return SubMatrixBlock<Dynamic, Dynamic>(derived(),
                                                startRow,
                                                startCol,
                                                blockRows,
                                                blockCols);
    }
    ConstSubMatrixBlock<Dynamic, Dynamic> block(Index startRow,
                                                Index startCol,
                                                int blockRows,
                                                int blockCols) const {
        return ConstSubMatrixBlock<Dynamic, Dynamic>(derived(),
                                                     startRow,
                                                     startCol,
                                                     blockRows,
                                                     blockCols);
    }
    template <int BlockRows, int BlockCols>
    SubMatrixBlock<BlockRows, BlockCols> block(Index startRow,
                                               Index startCol) {
        return SubMatrixBlock<BlockRows, BlockCols>(derived(),
                                                    startRow,
                                                    startCol);
    }
    template <int BlockRows, int BlockCols>
    ConstSubMatrixBlock<BlockRows, BlockCols> block(Index startRow,
                                                    Index startCol) const {
        return ConstSubMatrixBlock<BlockRows, BlockCols>(derived(),
                                                         startRow,
                                                         startCol);
    }
    SubMatrixBlock<Dynamic, Dynamic> topLeftCorner(int blockRows,
                                                   int blockCols) {
        return block(0, 0, blockRows, blockCols);
    }
    ConstSubMatrixBlock<Dynamic, Dynamic> topLeftCorner(int blockRows,
                                                        int blockCols) const {
        return block(0, 0, blockRows, blockCols);
    }
    template <int BlockRows, int BlockCols>
    SubMatrixBlock<BlockRows, BlockCols> topLeftCorner() {
        return block<BlockRows, BlockCols>(0, 0);
    }
    template <int BlockRows, int BlockCols>
    ConstSubMatrixBlock<BlockRows, BlockCols> topLeftCorner() const {
        return block<BlockRows, BlockCols>(0, 0);
    }
    SubMatrixBlock<Dynamic, Dynamic> topRightCorner(int blockRows,
                                                    int blockCols) {
        return block(0, cols()-blockCols, blockRows, blockCols);
    }
    ConstSubMatrixBlock<Dynamic, Dynamic> topRightCorner(int blockRows,
                                                         int blockCols) const {
        return block(0, cols()-blockCols, blockRows, blockCols);
    }
    template <int BlockRows, int BlockCols>
    SubMatrixBlock<BlockRows, BlockCols> topRightCorner() {
        return block<BlockRows, BlockCols>(0, cols()-BlockCols);
    }
    template <int BlockRows, int BlockCols>
    ConstSubMatrixBlock<BlockRows, BlockCols> topRightCorner() const {
        return block<BlockRows, BlockCols>(0, cols()-BlockCols);
    }
    SubMatrixBlock<Dynamic, Dynamic> bottomLeftCorner(int blockRows,
                                                      int blockCols) {
        return block(rows()-blockRows, 0, blockRows, blockCols);
    }
    ConstSubMatrixBlock<Dynamic, Dynamic>
    bottomLeftCorner(int blockRows, int blockCols) const {
        return block(rows()-blockRows, 0, blockRows, blockCols);
    }
    template <int BlockRows, int BlockCols>
    SubMatrixBlock<BlockRows, BlockCols> bottomLeftCorner() {
        return block<BlockRows, BlockCols>(rows()-BlockRows, 0);
    }
    template <int BlockRows, int BlockCols>
    ConstSubMatrixBlock<BlockRows, BlockCols> bottomLeftCorner() const {
        return block<BlockRows, BlockCols>(rows()-BlockRows, 0);
    }
    SubMatrixBlock<Dynamic, Dynamic> bottomRightCorner(int blockRows,
                                                       int blockCols) {
        return block(rows()-blockRows,
                     cols()-blockCols,
                     blockRows,
                     blockCols);
    }
    ConstSubMatrixBlock<Dynamic, Dynamic>
    bottomRightCorner(int blockRows, int blockCols) const {
        return block(rows()-blockRows,
                     cols()-blockCols,
                     blockRows,
                     blockCols);
    }
    template <int BlockRows, int BlockCols>
    SubMatrixBlock<BlockRows, BlockCols> bottomRightCorner() {
        return block<BlockRows, BlockCols>(rows()-BlockRows,
                                           cols()-BlockCols);
    }
    template <int BlockRows, int BlockCols>
    ConstSubMatrixBlock<BlockRows, BlockCols> bottomRightCorner() const {
        return block<BlockRows, BlockCols>(rows()-BlockRows,
                                           cols()-BlockCols);
    }
private:
    SubMatrixBlock<Dynamic, Cols> anyRows(int startRow, int blockRows) {
        return SubMatrixBlock<Dynamic, Cols>(derived(),
                                             startRow,
                                             0,
                                             blockRows,
                                             Cols);
    }
    ConstSubMatrixBlock<Dynamic, Cols> anyRows(int startRow,
                                               int blockRows) const {
        return ConstSubMatrixBlock<Dynamic, Cols>(
            derived(), startRow, 0, blockRows, Cols);
    }
    template <int BlockRows>
    SubMatrixBlock<BlockRows, Cols> anyRows(int startRow) {
        return SubMatrixBlock<BlockRows, Cols>(derived(),
                                               startRow,
                                               0,
                                               BlockRows,
                                               Cols);
    }
    template <int BlockRows>
    ConstSubMatrixBlock<BlockRows, Cols> anyRows(int startRow) const {
        return ConstSubMatrixBlock<BlockRows, Cols>(derived(),
                                                    startRow,
                                                    0,
                                                    BlockRows,
                                                    Cols);
    }
public:
    // Row and column accessors
    SubMatrixBlock<1, Cols> row(Index i) {
        return anyRows<1>(i);
    }
    ConstSubMatrixBlock<1, Cols> row(Index i) const {
        return anyRows<1>(i);
    }
    SubMatrixBlock<Dynamic, Cols> topRows(int blockRows) {
        return anyRows(0, blockRows);
    }
    ConstSubMatrixBlock<Dynamic, Cols> topRows(int blockRows) const {
        return anyRows(0, blockRows);
    }
    template <int BlockRows>
    SubMatrixBlock<BlockRows, Cols> topRows() {
        return anyRows<BlockRows>(0);
    }
    template <int BlockRows>
    ConstSubMatrixBlock<BlockRows, Cols> topRows() const {
        return anyRows<BlockRows>(0);
    }
    SubMatrixBlock<Dynamic, Cols> bottomRows(int blockRows) {
        return anyRows(rows()-blockRows, blockRows);
    }
    ConstSubMatrixBlock<Dynamic, Cols> bottomRows(int blockRows) const {
        return anyRows(rows()-blockRows, blockRows);
    }
    template <int BlockRows>
    SubMatrixBlock<BlockRows, Cols> bottomRows() {
        return anyRows<BlockRows>(rows()-BlockRows);
    }
    template <int BlockRows>
    ConstSubMatrixBlock<BlockRows, Cols> bottomRows() const {
        return anyRows<BlockRows>(rows()-BlockRows);
    }
private:
    SubMatrixBlock<Rows, Dynamic> anyCols(int startCol, int blockCols) {
        return SubMatrixBlock<Rows, Dynamic>(
            derived(), 0, startCol, Rows, blockCols);
    }
    ConstSubMatrixBlock<Rows, Dynamic> anyCols(int startCol,
                                               int blockCols) const {
        return ConstSubMatrixBlock<Rows, Dynamic>(
            derived(), 0, startCol, Rows, blockCols);
    }
    template <int BlockCols>
    SubMatrixBlock<Rows, BlockCols> anyCols(int startCol) {
        return SubMatrixBlock<Rows, BlockCols>(
            derived(), 0, startCol, Rows, BlockCols);
    }
    template <int BlockCols>
    ConstSubMatrixBlock<Rows, BlockCols> anyCols(int startCol) const {
        return ConstSubMatrixBlock<Rows, BlockCols>(
            derived(), 0, startCol, Rows, BlockCols);
    }
public:
    SubMatrixBlock<Rows, 1> col(Index j) {
        return anyCols<1>(j);
    }
    ConstSubMatrixBlock<Rows, 1> col(Index j) const {
        return anyCols<1>(j);
    }
    SubMatrixBlock<Rows, Dynamic> leftCols(int blockCols) {
        return anyCols(0, blockCols);
    }
    ConstSubMatrixBlock<Rows, Dynamic> leftCols(int blockCols) const {
        return anyCols(0, blockCols);
    }
    template <int BlockCols>
    SubMatrixBlock<Rows, BlockCols> leftCols() {
        return anyCols<BlockCols>(0);
    }
    template <int BlockCols>
    ConstSubMatrixBlock<Rows, BlockCols> leftCols() const {
        return anyCols<BlockCols>(0);
    }
    SubMatrixBlock<Rows, Dynamic> rightCols(int blockCols) {
        return anyCols(cols()-blockCols, blockCols);
    }
    ConstSubMatrixBlock<Rows, Dynamic> rightCols(int blockCols) const {
        return anyCols(cols()-blockCols, blockCols);
    }
    template <int BlockCols>
    SubMatrixBlock<Rows, BlockCols> rightCols() {
        return anyCols<BlockCols>(cols()-BlockCols);
    }
    template <int BlockCols>
    ConstSubMatrixBlock<Rows, BlockCols> rightCols() const {
        return anyCols<BlockCols>(cols()-BlockCols);
    }

    // Sub-block extractors (/!\ EAGEN EXTENSION /!\)
    //
    // Same idea as the sub-vector extractors above, but for matrix blocks.
    //
    // TODO: If this works out, extend to other operations
    //
    template <int Rows, int Cols>
    using SubMatrix = Matrix<Scalar, Rows, Cols>;
    template <int BlockRows, int BlockCols>
    SubMatrix<BlockRows, BlockCols> extractBlock(Index startRow,
                                                 Index startCol) const {
        if constexpr ((Rows != Dynamic) && (Cols != Dynamic)) {
            // Efficient static-sized version
            SubMatrix<BlockRows, BlockCols> result;
            for (Index col = 0; col < BlockCols; ++col) {
                for (Index row = 0; row < BlockRows; ++row) {
                    result.coeffRef(row, col) = coeff(row + startRow,
                                                      col + startCol);
                }
            }
            return result;
        } else {
            // Leave dynamic-sized matrices to Eigen
            return SubMatrix<BlockRows, BlockCols>(
                derivedInner().template block<BlockRows, BlockCols>(startRow,
                                                                    startCol)
            );
        }
    }
    template <int BlockRows, int BlockCols>
    SubMatrix<BlockRows, BlockCols> extractTopLeftCorner() const {
        return extractBlock<BlockRows, BlockCols>(0, 0);
    }
    RowVector<Scalar, Cols> extractRow(Index i) const {
        return extractBlock<1, Cols>(i, 0);
    }
    Vector<Scalar, Rows> extractCol(Index j) const {
        return extractBlock<Rows, 1>(0, j);
    }

    // /!\ UNDOCUMENTED /!\ Scalar cast
    template <typename NewScalarType>
    Matrix<NewScalarType, Rows, Cols> cast() const {
        return Matrix<NewScalarType, Rows, Cols>(
            derivedInner().template cast<NewScalarType>()
        );
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

private:
    static RealScalar dummy_precision() {
        return ScalarTraits::dummy_precision();
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
