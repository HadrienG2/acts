// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once


/// Key for type-erased data tuple comparisons
///
/// This container can be iteratively built out of a tuple of typed data. It is
/// usable as an associative container key, with the important property that two
/// DataKeys which were built out of identically typed data will compare equal
/// if and only if the data that was used to build them is effectively equal.
///
/// By "effectively equal", we mean that the data is equal according to a slight
/// modification of the standard C++ equality operator semantics which enforces
/// total ordering, i.e. x == x for any x. This is a philosophical deviation
/// from IEEE-754 equality, which C++ normally follows, when it comes to NaNs.
/// IEEE-754 is more concerned about equality having no false positives, whereas
/// in this program we are most concerned about it having no false negatives...
///
class TupleKey {
public:
  /// Set up key storage
  DataKey() = default;

  /// Record key material for a new column of data
  template <typename T>
  void addColumn(const T&); // TODO: Implement for all supported column types

  /// Clear key material so the underlying storage can be reused
  void clear() {
    m_keyBytes.clear();
  }

  // TODO: Add whatever's needed to make this type suitable for use as an
  //       unordered_map key. For hashing, abuse the implementation of std::hash
  //       for std::string.

private:
  /// This byte stream is the concatenation of one sub-stream per column.
  ///
  /// For each column, we turn the input data into bytes by canonicalizing it,
  /// that is to say, making sure that all data which is effectively equal turns
  /// into the same stream of bytes:
  ///
  /// - For integers, enums and booleans, we can just do a memcpy, since such
  ///   data compares equal if and only if the inner bytes are equal.
  /// - TODO: Check what needs to be done for floats.
  /// - But if we ever were to support UTF-8 strings, for example, we would need
  ///   to apply Unicode normalization, as in UTF-8 there are multiple byte
  ///   sequences that are considered equivalent for legacy reasons.
  /// - For data that contains pointers, the data behind those pointers must be
  ///   recursively serialized into the byte stream, otherwise TupleKey equality
  ///   would have false negatives (only match for shallow, not deep equality).
  /// - If the output length varies from one value of type T to another, then
  ///   the length of the sub-stream must be appended at the end. This ensures
  ///   that we never end up in the situation where a byte stream [ 1 ] [ 2 3 ]
  ///   compares equal to another byte stream [ 1 2 ] [ 3 ].
  ///
  /// Note that for the currently intended use case of TupleKey, one does _not_
  /// need to be able to go back from TupleKey to the input data.
  ///
  std::vector<char> m_keyBytes;
};
