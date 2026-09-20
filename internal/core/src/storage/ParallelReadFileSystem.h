// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <arrow/filesystem/filesystem.h>
#include <arrow/io/interfaces.h>

#include <cstdint>
#include <memory>
#include <string>

namespace milvus::storage {

// How a single large read is turned into several concurrent ranged reads.
//
// Parquet asks its file for one contiguous byte range per (coalesced) column
// chunk group. Arrow never splits such a range further -- CoalesceReadRanges
// only stops merging -- so one 30MB column chunk is one GetObject on the
// calling thread, and a reader that walks its files and ranges in order ends
// up with exactly one request in flight. Splitting the range restores the
// concurrency without changing how many bytes a read round holds.
struct ParallelReadConfig {
    // Reads larger than this are split into ceil(size / split_size_bytes)
    // parts. 0 disables splitting.
    int64_t split_size_bytes = 0;
    // Threads serving the parts. <= 0 disables splitting. The pool is separate
    // from arrow's IO pool on purpose: a split read is issued from a thread
    // that then waits for its parts, so sharing one pool could starve.
    int parallelism = 0;
};

// Applies the process-wide parallel read configuration. Resizes the pool when
// it already exists; a non-positive parallelism or split size turns splitting
// off for readers opened afterwards.
void
ConfigureParallelRead(int64_t split_size_bytes, int parallelism);

ParallelReadConfig
GetParallelReadConfig();

// Wraps fs so that OpenInputFile hands back a file whose ReadAt splits large
// reads across the parallel read pool. Returns fs unchanged when fs is null or
// splitting is disabled, so callers can wrap unconditionally.
std::shared_ptr<arrow::fs::FileSystem>
WrapParallelRead(std::shared_ptr<arrow::fs::FileSystem> fs);

// Wraps one already-open file, for callers that hold a file rather than a
// filesystem. Returns file unchanged when splitting is disabled.
std::shared_ptr<arrow::io::RandomAccessFile>
WrapParallelRead(std::shared_ptr<arrow::io::RandomAccessFile> file);

}  // namespace milvus::storage
