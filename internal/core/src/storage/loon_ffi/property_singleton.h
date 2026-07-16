// Copyright 2023 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <utility>

#include "common/type_c.h"
#include "milvus-storage/properties.h"
#include "util.h"

namespace milvus::storage {

class LoonFFIPropertiesSingleton {
 private:
    LoonFFIPropertiesSingleton() = default;

 public:
    static LoonFFIPropertiesSingleton&
    GetInstance() {
        static LoonFFIPropertiesSingleton instance;
        return instance;
    }

    void
    Init(CStorageConfig c_storage_config) {
        std::unique_lock lck(mutex_);

        if (properties_ == nullptr) {
            properties_ =
                MakeInternalPropertiesFromStorageConfig(c_storage_config);
            ApplyArrowReaderConfig(*properties_);
        }
    }

    void
    Init(const char* root_path) {
        std::unique_lock lck(mutex_);

        if (properties_ == nullptr) {
            properties_ = MakeInternalLocalProperies(root_path);
            ApplyArrowReaderConfig(*properties_);
        }
    }

    void
    SetArrowReaderConfig(int64_t hole_size_limit_bytes,
                         int64_t range_size_limit_bytes) {
        arrow_reader_hole_size_limit_bytes_.store(hole_size_limit_bytes,
                                                  std::memory_order_relaxed);
        arrow_reader_range_size_limit_bytes_.store(range_size_limit_bytes,
                                                   std::memory_order_relaxed);
        std::unique_lock lck(mutex_);
        if (properties_ != nullptr) {
            auto properties =
                std::make_shared<milvus_storage::api::Properties>(*properties_);
            ApplyArrowReaderConfig(*properties);
            properties_ = std::move(properties);
        }
    }

    std::shared_ptr<milvus_storage::api::Properties>
    GetProperties() const {
        std::shared_lock lck(mutex_);
        return properties_;
    }

    // Sets the per-round read window (loon's reader.record_batch_max_size)
    // applied to index-build manifest reads. 0 keeps loon's built-in default
    // (32MB, which admits only one 64MB-class row group per prefetch round
    // and therefore serializes the raw-data download to one S3 GET at a
    // time). Applied per task via ApplyIndexBuildReadWindow — deliberately
    // NOT part of ApplyArrowReaderConfig so query-node load paths keep
    // their small default window.
    void
    SetIndexBuildReadWindow(int64_t window_bytes) {
        index_build_read_window_bytes_.store(window_bytes,
                                             std::memory_order_relaxed);
    }

    int64_t
    GetIndexBuildReadWindow() const {
        return index_build_read_window_bytes_.load(std::memory_order_relaxed);
    }

    // Stamps the index-build read window into `properties` when configured
    // (> 0). Lock-free; no-op when unset so loon's default applies.
    void
    ApplyIndexBuildReadWindow(
        milvus_storage::api::Properties& properties) const {
        auto window =
            index_build_read_window_bytes_.load(std::memory_order_relaxed);
        if (window <= 0) {
            return;
        }
        milvus_storage::api::SetValue(properties,
                                      PROPERTY_READER_RECORD_BATCH_MAX_SIZE,
                                      std::to_string(window).c_str());
    }

    // Applies the configured arrow reader prebuffer limits to `properties`.
    // Lock-free (reads atomics only) so it is safe to call from
    // MakeInternalPropertiesFromStorageConfig even while this singleton
    // holds mutex_ (Init builds its cached map through that same helper).
    // A value of 0 means "keep the arrow default" downstream.
    void
    ApplyArrowReaderConfig(milvus_storage::api::Properties& properties) const {
        milvus_storage::api::SetValue(
            properties,
            PROPERTY_READER_PARQUET_PREBUFFER_HOLE_SIZE_LIMIT,
            std::to_string(arrow_reader_hole_size_limit_bytes_.load(
                               std::memory_order_relaxed))
                .c_str());
        milvus_storage::api::SetValue(
            properties,
            PROPERTY_READER_PARQUET_PREBUFFER_RANGE_SIZE_LIMIT,
            std::to_string(arrow_reader_range_size_limit_bytes_.load(
                               std::memory_order_relaxed))
                .c_str());
    }

 private:
    mutable std::shared_mutex mutex_;
    std::shared_ptr<milvus_storage::api::Properties> properties_ = nullptr;
    std::atomic<int64_t> arrow_reader_hole_size_limit_bytes_{0};
    std::atomic<int64_t> arrow_reader_range_size_limit_bytes_{0};
    std::atomic<int64_t> index_build_read_window_bytes_{0};
};

}  // namespace milvus::storage
