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

#include "storage/ParallelReadFileSystem.h"

#include <arrow/buffer.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/util/future.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/util/thread_pool.h>

#include <algorithm>
#include <mutex>
#include <utility>
#include <vector>

#include "log/Log.h"

namespace milvus::storage {
namespace {

std::mutex parallel_read_mutex;
ParallelReadConfig parallel_read_config;
std::shared_ptr<arrow::internal::ThreadPool> parallel_read_pool;

// Snapshot of the configuration a reader is opened with. A reader keeps the
// pool it was opened with for its lifetime, so a later resize cannot pull the
// pool out from under an in-flight read.
struct ParallelReadContext {
    int64_t split_size_bytes = 0;
    std::shared_ptr<arrow::internal::ThreadPool> pool;

    bool
    enabled() const {
        return split_size_bytes > 0 && pool != nullptr;
    }
};

ParallelReadContext
CurrentContext() {
    std::lock_guard<std::mutex> lock(parallel_read_mutex);
    ParallelReadContext ctx;
    ctx.split_size_bytes = parallel_read_config.split_size_bytes;
    if (parallel_read_config.parallelism > 0) {
        ctx.pool = parallel_read_pool;
    }
    return ctx;
}

// A file whose ReadAt splits a large range into parts fetched concurrently.
// Everything else is delegated: the position-based methods keep using the
// inner file's own cursor, so sequential reads behave exactly as before.
class ParallelReadInputFile : public arrow::io::RandomAccessFile {
 public:
    ParallelReadInputFile(std::shared_ptr<arrow::io::RandomAccessFile> inner,
                          ParallelReadContext ctx)
        : inner_(std::move(inner)), ctx_(std::move(ctx)) {
    }

    arrow::Status
    Close() override {
        return inner_->Close();
    }

    bool
    closed() const override {
        return inner_->closed();
    }

    arrow::Result<int64_t>
    Tell() const override {
        return inner_->Tell();
    }

    arrow::Status
    Seek(int64_t position) override {
        return inner_->Seek(position);
    }

    arrow::Result<int64_t>
    GetSize() override {
        return inner_->GetSize();
    }

    arrow::Result<int64_t>
    Read(int64_t nbytes, void* out) override {
        return inner_->Read(nbytes, out);
    }

    arrow::Result<std::shared_ptr<arrow::Buffer>>
    Read(int64_t nbytes) override {
        return inner_->Read(nbytes);
    }

    arrow::Result<int64_t>
    ReadAt(int64_t position, int64_t nbytes, void* out) override {
        if (!ShouldSplit(nbytes)) {
            return inner_->ReadAt(position, nbytes, out);
        }
        return ReadParts(position, nbytes, static_cast<uint8_t*>(out));
    }

    arrow::Result<std::shared_ptr<arrow::Buffer>>
    ReadAt(int64_t position, int64_t nbytes) override {
        if (!ShouldSplit(nbytes)) {
            return inner_->ReadAt(position, nbytes);
        }
        ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateBuffer(nbytes));
        ARROW_ASSIGN_OR_RAISE(
            auto read, ReadParts(position, nbytes, buffer->mutable_data()));
        if (read < nbytes) {
            // Short read at end of file: hand back only what exists, matching
            // the contract of the underlying ReadAt.
            return arrow::SliceBuffer(
                std::shared_ptr<arrow::Buffer>(std::move(buffer)), 0, read);
        }
        return std::shared_ptr<arrow::Buffer>(std::move(buffer));
    }

 private:
    bool
    ShouldSplit(int64_t nbytes) const {
        return ctx_.enabled() && nbytes > ctx_.split_size_bytes;
    }

    // Reads [position, position + nbytes) into out through several concurrent
    // ranged reads. The first part runs on the calling thread so a saturated
    // pool cannot stall the read completely.
    arrow::Result<int64_t>
    ReadParts(int64_t position, int64_t nbytes, uint8_t* out) {
        const int64_t part_size = ctx_.split_size_bytes;
        const int64_t num_parts = (nbytes + part_size - 1) / part_size;

        std::vector<arrow::Future<int64_t>> futures;
        futures.reserve(num_parts - 1);
        for (int64_t i = 1; i < num_parts; ++i) {
            const int64_t offset = i * part_size;
            const int64_t length = std::min(part_size, nbytes - offset);
            auto inner = inner_;
            auto submitted =
                ctx_.pool->Submit([inner, position, offset, length, out]()
                                      -> arrow::Result<int64_t> {
                    return inner->ReadAt(
                        position + offset, length, out + offset);
                });
            if (!submitted.ok()) {
                // The pool refused the task (shutting down). Fall back to
                // reading the rest on this thread rather than failing.
                for (int64_t j = i; j < num_parts; ++j) {
                    const int64_t fb_offset = j * part_size;
                    const int64_t fb_length =
                        std::min(part_size, nbytes - fb_offset);
                    ARROW_RETURN_NOT_OK(inner_->ReadAt(
                        position + fb_offset, fb_length, out + fb_offset));
                }
                break;
            }
            futures.emplace_back(submitted.MoveValueUnsafe());
        }

        int64_t total = 0;
        auto head = inner_->ReadAt(position, std::min(part_size, nbytes), out);
        arrow::Status status = head.status();
        if (head.ok()) {
            total += head.ValueOrDie();
        }
        // Every submitted part must be waited on even after a failure: the
        // tasks write into out, which is owned by the caller.
        for (auto& future : futures) {
            auto result = future.result();
            if (!result.ok()) {
                if (status.ok()) {
                    status = result.status();
                }
                continue;
            }
            total += result.ValueOrDie();
        }
        ARROW_RETURN_NOT_OK(status);
        return total;
    }

    std::shared_ptr<arrow::io::RandomAccessFile> inner_;
    ParallelReadContext ctx_;
};

// Delegating filesystem whose only behavior change is the files it opens.
class ParallelReadFileSystem : public arrow::fs::FileSystem {
 public:
    ParallelReadFileSystem(std::shared_ptr<arrow::fs::FileSystem> inner,
                           ParallelReadContext ctx)
        : arrow::fs::FileSystem(inner->io_context()),
          inner_(std::move(inner)),
          ctx_(std::move(ctx)) {
    }

    std::string
    type_name() const override {
        return inner_->type_name();
    }

    const std::shared_ptr<arrow::fs::FileSystem>&
    inner() const {
        return inner_;
    }

    bool
    Equals(const arrow::fs::FileSystem& other) const override {
        // type_name() reports the wrapped filesystem's name, so a same-named
        // other may be an unwrapped filesystem: compare against whichever
        // filesystem actually does the IO.
        const auto* other_wrapper =
            dynamic_cast<const ParallelReadFileSystem*>(&other);
        if (other_wrapper != nullptr) {
            return inner_->Equals(other_wrapper->inner());
        }
        return inner_->Equals(other);
    }

    arrow::Result<std::string>
    NormalizePath(std::string path) override {
        return inner_->NormalizePath(std::move(path));
    }

    arrow::Result<arrow::fs::FileInfo>
    GetFileInfo(const std::string& path) override {
        return inner_->GetFileInfo(path);
    }

    arrow::Result<arrow::fs::FileInfoVector>
    GetFileInfo(const arrow::fs::FileSelector& select) override {
        return inner_->GetFileInfo(select);
    }

    arrow::Status
    CreateDir(const std::string& path, bool recursive) override {
        return inner_->CreateDir(path, recursive);
    }

    arrow::Status
    DeleteDir(const std::string& path) override {
        return inner_->DeleteDir(path);
    }

    arrow::Status
    DeleteDirContents(const std::string& path, bool missing_dir_ok) override {
        return inner_->DeleteDirContents(path, missing_dir_ok);
    }

    arrow::Status
    DeleteRootDirContents() override {
        return inner_->DeleteRootDirContents();
    }

    arrow::Status
    DeleteFile(const std::string& path) override {
        return inner_->DeleteFile(path);
    }

    arrow::Status
    Move(const std::string& src, const std::string& dest) override {
        return inner_->Move(src, dest);
    }

    arrow::Status
    CopyFile(const std::string& src, const std::string& dest) override {
        return inner_->CopyFile(src, dest);
    }

    arrow::Result<std::shared_ptr<arrow::io::InputStream>>
    OpenInputStream(const std::string& path) override {
        return inner_->OpenInputStream(path);
    }

    arrow::Result<std::shared_ptr<arrow::io::InputStream>>
    OpenInputStream(const arrow::fs::FileInfo& info) override {
        return inner_->OpenInputStream(info);
    }

    arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>>
    OpenInputFile(const std::string& path) override {
        ARROW_ASSIGN_OR_RAISE(auto file, inner_->OpenInputFile(path));
        return Wrap(std::move(file));
    }

    arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>>
    OpenInputFile(const arrow::fs::FileInfo& info) override {
        ARROW_ASSIGN_OR_RAISE(auto file, inner_->OpenInputFile(info));
        return Wrap(std::move(file));
    }

    arrow::Result<std::shared_ptr<arrow::io::OutputStream>>
    OpenOutputStream(const std::string& path,
                     const std::shared_ptr<const arrow::KeyValueMetadata>&
                         metadata) override {
        return inner_->OpenOutputStream(path, metadata);
    }

    arrow::Result<std::shared_ptr<arrow::io::OutputStream>>
    OpenAppendStream(const std::string& path,
                     const std::shared_ptr<const arrow::KeyValueMetadata>&
                         metadata) override {
        return inner_->OpenAppendStream(path, metadata);
    }

 private:
    std::shared_ptr<arrow::io::RandomAccessFile>
    Wrap(std::shared_ptr<arrow::io::RandomAccessFile> file) const {
        return std::make_shared<ParallelReadInputFile>(std::move(file), ctx_);
    }

    std::shared_ptr<arrow::fs::FileSystem> inner_;
    ParallelReadContext ctx_;
};

}  // namespace

void
ConfigureParallelRead(int64_t split_size_bytes, int parallelism) {
    std::lock_guard<std::mutex> lock(parallel_read_mutex);
    parallel_read_config.split_size_bytes =
        std::max<int64_t>(split_size_bytes, 0);
    parallel_read_config.parallelism = std::max(parallelism, 0);

    if (parallel_read_config.parallelism <= 0) {
        // Keep the pool: readers opened earlier still hold it, and creating a
        // new one on re-enable is cheap compared to tearing this one down.
        LOG_INFO("parallel read disabled, splitSizeBytes={}, parallelism={}",
                 parallel_read_config.split_size_bytes,
                 parallel_read_config.parallelism);
        return;
    }
    if (parallel_read_pool == nullptr) {
        auto pool_result =
            arrow::internal::ThreadPool::Make(parallel_read_config.parallelism);
        if (!pool_result.ok()) {
            LOG_WARN("failed to create parallel read pool with {} threads: {}",
                     parallel_read_config.parallelism,
                     pool_result.status().ToString());
            parallel_read_config.parallelism = 0;
            return;
        }
        parallel_read_pool = pool_result.MoveValueUnsafe();
    } else {
        auto status =
            parallel_read_pool->SetCapacity(parallel_read_config.parallelism);
        if (!status.ok()) {
            LOG_WARN("failed to resize parallel read pool to {}: {}",
                     parallel_read_config.parallelism,
                     status.ToString());
        }
    }
    LOG_INFO("parallel read configured, splitSizeBytes={}, parallelism={}",
             parallel_read_config.split_size_bytes,
             parallel_read_config.parallelism);
}

ParallelReadConfig
GetParallelReadConfig() {
    std::lock_guard<std::mutex> lock(parallel_read_mutex);
    return parallel_read_config;
}

std::shared_ptr<arrow::fs::FileSystem>
WrapParallelRead(std::shared_ptr<arrow::fs::FileSystem> fs) {
    if (fs == nullptr) {
        return fs;
    }
    auto ctx = CurrentContext();
    if (!ctx.enabled()) {
        return fs;
    }
    return std::make_shared<ParallelReadFileSystem>(std::move(fs),
                                                    std::move(ctx));
}

std::shared_ptr<arrow::io::RandomAccessFile>
WrapParallelRead(std::shared_ptr<arrow::io::RandomAccessFile> file) {
    if (file == nullptr) {
        return file;
    }
    auto ctx = CurrentContext();
    if (!ctx.enabled()) {
        return file;
    }
    return std::make_shared<ParallelReadInputFile>(std::move(file),
                                                   std::move(ctx));
}

}  // namespace milvus::storage
