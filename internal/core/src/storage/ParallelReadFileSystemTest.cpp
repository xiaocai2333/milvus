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

#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "storage/ParallelReadFileSystem.h"

namespace {

using milvus::storage::ConfigureParallelRead;
using milvus::storage::GetParallelReadConfig;
using milvus::storage::WrapParallelRead;

constexpr int64_t kSplitSize = 1024;

// A file that records how its ReadAt is called: how many times, and how many
// calls were in flight at once. Each read sleeps briefly so overlapping calls
// are observable rather than racing to completion one by one.
class CountingFile : public arrow::io::RandomAccessFile {
 public:
    explicit CountingFile(std::string data,
                          arrow::Status fail_status = arrow::Status::OK())
        : data_(std::move(data)), fail_status_(std::move(fail_status)) {
    }

    arrow::Status
    Close() override {
        closed_ = true;
        return arrow::Status::OK();
    }

    bool
    closed() const override {
        return closed_;
    }

    arrow::Result<int64_t>
    Tell() const override {
        return position_;
    }

    arrow::Status
    Seek(int64_t position) override {
        position_ = position;
        return arrow::Status::OK();
    }

    arrow::Result<int64_t>
    GetSize() override {
        return static_cast<int64_t>(data_.size());
    }

    arrow::Result<int64_t>
    Read(int64_t nbytes, void* out) override {
        ARROW_ASSIGN_OR_RAISE(auto read, ReadAt(position_, nbytes, out));
        position_ += read;
        return read;
    }

    arrow::Result<std::shared_ptr<arrow::Buffer>>
    Read(int64_t nbytes) override {
        ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(position_, nbytes));
        position_ += buffer->size();
        return buffer;
    }

    arrow::Result<int64_t>
    ReadAt(int64_t position, int64_t nbytes, void* out) override {
        const int64_t in_flight = ++in_flight_;
        int64_t previous_max = max_in_flight_.load();
        while (in_flight > previous_max &&
               !max_in_flight_.compare_exchange_weak(previous_max, in_flight)) {
        }
        ++calls_;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        --in_flight_;

        if (!fail_status_.ok()) {
            return fail_status_;
        }
        const int64_t size = static_cast<int64_t>(data_.size());
        if (position >= size) {
            return 0;
        }
        const int64_t length = std::min(nbytes, size - position);
        std::memcpy(out, data_.data() + position, length);
        return length;
    }

    arrow::Result<std::shared_ptr<arrow::Buffer>>
    ReadAt(int64_t position, int64_t nbytes) override {
        ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateBuffer(nbytes));
        ARROW_ASSIGN_OR_RAISE(auto read,
                              ReadAt(position, nbytes, buffer->mutable_data()));
        return arrow::SliceBuffer(
            std::shared_ptr<arrow::Buffer>(std::move(buffer)), 0, read);
    }

    int64_t
    calls() const {
        return calls_;
    }

    int64_t
    max_in_flight() const {
        return max_in_flight_;
    }

 private:
    std::string data_;
    arrow::Status fail_status_;
    int64_t position_ = 0;
    bool closed_ = false;
    std::atomic<int64_t> calls_{0};
    std::atomic<int64_t> in_flight_{0};
    std::atomic<int64_t> max_in_flight_{0};
};

std::string
MakeData(int64_t size) {
    std::string data(size, '\0');
    for (int64_t i = 0; i < size; ++i) {
        data[i] = static_cast<char>(i % 251);
    }
    return data;
}

class ParallelReadTest : public ::testing::Test {
 protected:
    void
    SetUp() override {
        saved_ = GetParallelReadConfig();
        ConfigureParallelRead(kSplitSize, 4);
    }

    void
    TearDown() override {
        ConfigureParallelRead(saved_.split_size_bytes, saved_.parallelism);
    }

    milvus::storage::ParallelReadConfig saved_;
};

TEST_F(ParallelReadTest, SplitsLargeReadAcrossThePool) {
    auto data = MakeData(kSplitSize * 8);
    auto inner = std::make_shared<CountingFile>(data);
    auto file = WrapParallelRead(inner);
    ASSERT_NE(file.get(), inner.get())
        << "an enabled config must wrap the file";

    auto result = file->ReadAt(0, static_cast<int64_t>(data.size()));
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    auto buffer = result.ValueOrDie();
    ASSERT_EQ(buffer->size(), static_cast<int64_t>(data.size()));
    EXPECT_EQ(std::memcmp(buffer->data(), data.data(), data.size()), 0)
        << "parts must land at their own offsets, in order";

    EXPECT_EQ(inner->calls(), 8) << "one request per split-sized part";
    EXPECT_GT(inner->max_in_flight(), 1)
        << "parts must be fetched concurrently, not one after another";
}

TEST_F(ParallelReadTest, ReadNotLargerThanSplitStaysOneRequest) {
    auto data = MakeData(kSplitSize);
    auto inner = std::make_shared<CountingFile>(data);
    auto file = WrapParallelRead(inner);

    auto result = file->ReadAt(0, kSplitSize);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result.ValueOrDie()->size(), kSplitSize);
    EXPECT_EQ(inner->calls(), 1);
    EXPECT_EQ(inner->max_in_flight(), 1);
}

TEST_F(ParallelReadTest, ShortReadAtEndOfFileIsTruncated) {
    auto data = MakeData(kSplitSize * 3 + 7);
    auto inner = std::make_shared<CountingFile>(data);
    auto file = WrapParallelRead(inner);

    // Ask for more than the file holds: the result must stop at the end of
    // the data instead of reporting the requested length.
    auto result = file->ReadAt(0, kSplitSize * 8);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    auto buffer = result.ValueOrDie();
    ASSERT_EQ(buffer->size(), static_cast<int64_t>(data.size()));
    EXPECT_EQ(std::memcmp(buffer->data(), data.data(), data.size()), 0);
}

TEST_F(ParallelReadTest, IntoCallerBufferMatchesSerialRead) {
    auto data = MakeData(kSplitSize * 5);
    auto inner = std::make_shared<CountingFile>(data);
    auto file = WrapParallelRead(inner);

    std::vector<char> out(data.size(), 0);
    auto result =
        file->ReadAt(0, static_cast<int64_t>(data.size()), out.data());
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result.ValueOrDie(), static_cast<int64_t>(data.size()));
    EXPECT_EQ(std::memcmp(out.data(), data.data(), data.size()), 0);
}

TEST_F(ParallelReadTest, PartFailureIsReported) {
    auto data = MakeData(kSplitSize * 4);
    auto inner = std::make_shared<CountingFile>(
        data, arrow::Status::IOError("object storage unavailable"));
    auto file = WrapParallelRead(inner);

    auto result = file->ReadAt(0, static_cast<int64_t>(data.size()));
    ASSERT_FALSE(result.ok());
    EXPECT_TRUE(result.status().IsIOError()) << result.status().ToString();
}

TEST_F(ParallelReadTest, DisabledConfigLeavesTheFileAlone) {
    ConfigureParallelRead(0, 4);
    auto inner = std::make_shared<CountingFile>(MakeData(kSplitSize * 4));
    EXPECT_EQ(WrapParallelRead(inner).get(), inner.get());

    ConfigureParallelRead(kSplitSize, 0);
    EXPECT_EQ(WrapParallelRead(inner).get(), inner.get());

    EXPECT_EQ(
        WrapParallelRead(std::shared_ptr<arrow::io::RandomAccessFile>()).get(),
        nullptr);
    EXPECT_EQ(WrapParallelRead(std::shared_ptr<arrow::fs::FileSystem>()).get(),
              nullptr);
}

TEST_F(ParallelReadTest, ConfigIsReadBack) {
    ConfigureParallelRead(4096, 3);
    auto config = GetParallelReadConfig();
    EXPECT_EQ(config.split_size_bytes, 4096);
    EXPECT_EQ(config.parallelism, 3);

    // Negative values are clamped to "off" rather than reaching the pool.
    ConfigureParallelRead(-1, -1);
    config = GetParallelReadConfig();
    EXPECT_EQ(config.split_size_bytes, 0);
    EXPECT_EQ(config.parallelism, 0);
}

}  // namespace
