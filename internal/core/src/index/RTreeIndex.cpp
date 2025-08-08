// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include "index/RTreeIndex.h"
#include <boost/filesystem.hpp>
#include "common/EasyAssert.h"
#include "common/FieldData.h"
#include "log/Log.h"
#include "index/Utils.h"
#include "index/Meta.h"
#include "storage/LocalChunkManagerSingleton.h"
#include "pb/schema.pb.h"

namespace milvus::index {

constexpr const char* TMP_RTREE_INDEX_PREFIX = "/tmp/milvus/rtree-index/";

// helper to check suffix
static inline bool
ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
               0;
}

template <typename T>
void
RTreeIndex<T>::InitForBuildIndex() {
    auto field =
        std::to_string(disk_file_manager_->GetFieldDataMeta().field_id);
    auto prefix = disk_file_manager_->GetIndexIdentifier();
    path_ = std::string(TMP_RTREE_INDEX_PREFIX) + prefix;
    boost::filesystem::create_directories(path_);

    std::string index_file_path = path_ + "/index_file";  // base path (no ext)

    if (boost::filesystem::exists(index_file_path + ".dat") ||
        boost::filesystem::exists(index_file_path + ".idx")) {
        PanicInfo(
            IndexBuildError, "build rtree index temp dir:{} not empty", path_);
    }
    LOG_INFO("build rtree index temp dir:{}", index_file_path);
    wrapper_ = std::make_shared<RTreeIndexWrapper>(index_file_path, true);
    LOG_INFO("build rtree index wrapper success");
}

template <typename T>
RTreeIndex<T>::RTreeIndex(const storage::FileManagerContext& ctx)
    : ScalarIndex<T>(RTREE_INDEX_TYPE),
      schema_(ctx.fieldDataMeta.field_schema) {
    mem_file_manager_ = std::make_shared<MemFileManager>(ctx);
    disk_file_manager_ = std::make_shared<DiskFileManager>(ctx);

    if (ctx.for_loading_index) {
        return;
    }
}

template <typename T>
RTreeIndex<T>::~RTreeIndex() {
    // Free wrapper explicitly to ensure files not being used
    wrapper_.reset();

    // Remove temporary directory if it exists
    if (!path_.empty()) {
        auto local_cm = storage::LocalChunkManagerSingleton::GetInstance()
                            .GetChunkManager();
        if (local_cm) {
            LOG_INFO("rtree index remove path:{}", path_);
            local_cm->RemoveDir(path_);
        }
    }
}

static std::string
GetFileName(const std::string& path) {
    auto pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

// Loading existing R-Tree index
// The config must contain "index_files" -> vector<string>
// Remote index objects will be downloaded to local disk via DiskFileManager,
// then RTreeIndexWrapper will load them.
template <typename T>
void
RTreeIndex<T>::Load(milvus::tracer::TraceContext ctx, const Config& config) {
    LOG_DEBUG("Load RTreeIndex with config {}", config.dump());

    auto index_files_opt =
        GetValueFromConfig<std::vector<std::string>>(config, "index_files");
    AssertInfo(index_files_opt.has_value(),
               "index file paths are empty when loading R-Tree index");

    auto files = index_files_opt.value();

    // 1. Extract potential null_offset file (optional)
    {
        auto it = std::find_if(
            files.begin(), files.end(), [](const std::string& file) {
                return GetFileName(file) == "index_null_offset";
            });
        if (it != files.end()) {
            std::vector<std::string> tmp{*it};
            files.erase(it);
            auto index_datas = mem_file_manager_->LoadIndexToMemory(tmp);
            BinarySet binary_set;
            AssembleIndexDatas(index_datas, binary_set);
        }
    }

    // 2. Ensure each file has full remote path. If only filename provided, prepend remote prefix.
    for (auto& f : files) {
        boost::filesystem::path p(f);
        if (!p.has_parent_path()) {
            auto remote_prefix = disk_file_manager_->GetRemoteIndexPrefix();
            f = remote_prefix + "/" + f;
        }
    }

    // 3. Cache remote index files to local disk.
    disk_file_manager_->CacheIndexToDisk(files);

    // 4. Determine local base path (without extension) for RTreeIndexWrapper.
    auto local_paths = disk_file_manager_->GetLocalFilePaths();
    AssertInfo(!local_paths.empty(),
               "RTreeIndex local files are empty after caching to disk");

    // Pick a .dat or .idx file explicitly; avoid meta or others.
    std::string base_path;
    for (const auto& p : local_paths) {
        if (ends_with(p, ".dat")) {
            base_path = p.substr(0, p.size() - 4);
            break;
        }
        if (ends_with(p, ".idx")) {
            base_path = p.substr(0, p.size() - 4);
            break;
        }
    }
    // Fallback: if not found, try meta json
    if (base_path.empty()) {
        for (const auto& p : local_paths) {
            if (ends_with(p, ".meta.json")) {
                base_path =
                    p.substr(0, p.size() - std::string(".meta.json").size());
                break;
            }
        }
    }
    // Final fallback: use the first path as-is
    if (base_path.empty()) {
        base_path = local_paths.front();
    }
    path_ = base_path;

    // 5. Instantiate wrapper and load.
    wrapper_ =
        std::make_shared<RTreeIndexWrapper>(path_, /*is_build_mode=*/false);
    wrapper_->load();

    total_num_rows_ = wrapper_->count();
    is_built_ = true;

    LOG_INFO(
        "Loaded R-Tree index from {} with {} rows", path_, total_num_rows_);
}

template <typename T>
void
RTreeIndex<T>::Build(const Config& config) {
    auto insert_files =
        GetValueFromConfig<std::vector<std::string>>(config, "insert_files");
    AssertInfo(insert_files.has_value(),
               "insert_files were empty for building RTree index");

    InitForBuildIndex();

    auto fill_factor =
        GetValueFromConfig<double>(config, FILL_FACTOR_KEY).value_or(0.8);
    auto index_cap =
        GetValueFromConfig<uint32_t>(config, INDEX_CAPACITY_KEY).value_or(100);
    auto leaf_cap =
        GetValueFromConfig<uint32_t>(config, LEAF_CAPACITY_KEY).value_or(100);
    auto variant_str =
        GetValueFromConfig<std::string>(config, R_TREE_VARIANT_KEY)
            .value_or("RSTAR");

    wrapper_->set_fill_factor(fill_factor);
    wrapper_->set_index_capacity(index_cap);
    wrapper_->set_leaf_capacity(leaf_cap);
    wrapper_->set_rtree_variant(variant_str);

    // load raw WKB data into memory
    auto field_datas =
        mem_file_manager_->CacheRawDataToMemory(insert_files.value());

    BuildWithFieldData(field_datas);

    // after build, mark built
    total_num_rows_ = wrapper_->count();
    is_built_ = true;
}

template <typename T>
void
RTreeIndex<T>::BuildWithFieldData(
    const std::vector<FieldDataPtr>& field_datas) {
    int64_t offset = 0;

    for (const auto& data : field_datas) {
        auto n = data->get_num_rows();
        for (int64_t i = 0; i < n; ++i) {
            // Neglect null for now
            if (schema_.nullable() && !data->is_valid(i)) {
                ++offset;
                continue;
            }
            // get wkb str from field data
            auto wkb_ptr = static_cast<const std::string*>(data->RawValue(i));
            // convert to uint8_t*
            const uint8_t* wkb =
                reinterpret_cast<const uint8_t*>(wkb_ptr->data());
            size_t len = wkb_ptr->size();

            // add to rtree index
            wrapper_->add_geometry(wkb, len, offset);
            ++offset;
        }
    }
}

template <typename T>
void
RTreeIndex<T>::finish() {
    if (wrapper_) {
        LOG_INFO("rtree index finish");
        wrapper_->finish();
    }
}

template <typename T>
IndexStatsPtr
RTreeIndex<T>::Upload(const Config& config) {
    // 1. Ensure all buffered data flushed to disk
    finish();

    // 2. Walk temp dir and register files to DiskFileManager
    boost::filesystem::path dir(path_);
    boost::filesystem::directory_iterator end_iter;

    for (boost::filesystem::directory_iterator it(dir); it != end_iter; ++it) {
        if (boost::filesystem::is_directory(*it)) {
            LOG_WARN("{} is a directory, skip", it->path().string());
            continue;
        }

        LOG_INFO("trying to add index file: {}", it->path().string());
        AssertInfo(disk_file_manager_->AddFile(it->path().string()),
                   "failed to add index file: {}",
                   it->path().string());
        LOG_INFO("index file: {} added", it->path().string());
    }

    // 3. Collect remote paths to size mapping
    auto remote_paths_to_size = disk_file_manager_->GetRemotePathsToFileSize();

    // 4. Assemble IndexStats result (no in-memory part for now)
    std::vector<SerializedIndexFileInfo> index_files;
    index_files.reserve(remote_paths_to_size.size());
    for (auto& kv : remote_paths_to_size) {
        index_files.emplace_back(kv.first, kv.second);
    }

    int64_t mem_size = mem_file_manager_->GetAddedTotalMemSize();  // likely 0
    int64_t file_size = disk_file_manager_->GetAddedTotalFileSize();

    return IndexStats::New(mem_size + file_size, std::move(index_files));
}

template <typename T>
BinarySet
RTreeIndex<T>::Serialize(const Config& config) {
    PanicInfo(ErrorCode::NotImplemented,
              "Serialize() is not yet supported for RTreeIndex");
    return {};
}

template <typename T>
void
RTreeIndex<T>::Load(const BinarySet& binary_set, const Config& config) {
    PanicInfo(ErrorCode::NotImplemented,
              "Load(BinarySet) is not yet supported for RTreeIndex");
}

template <typename T>
void
RTreeIndex<T>::Build(size_t n, const T* values, const bool* valid_data) {
    // Generic Build by value array is not required for RTree at the moment.
    PanicInfo(ErrorCode::NotImplemented,
              "Build(size_t, values, valid) not supported for RTreeIndex");
}

template <typename T>
const TargetBitmap
RTreeIndex<T>::In(size_t n, const T* values) {
    PanicInfo(ErrorCode::NotImplemented, "In() not supported for RTreeIndex");
    return {};
}

template <typename T>
const TargetBitmap
RTreeIndex<T>::IsNull() {
    PanicInfo(ErrorCode::NotImplemented,
              "IsNull() not supported for RTreeIndex");
    return {};
}

template <typename T>
const TargetBitmap
RTreeIndex<T>::IsNotNull() {
    PanicInfo(ErrorCode::NotImplemented,
              "IsNotNull() not supported for RTreeIndex");
    return {};
}

template <typename T>
const TargetBitmap
RTreeIndex<T>::InApplyFilter(size_t n,
                             const T* values,
                             const std::function<bool(size_t)>& filter) {
    PanicInfo(ErrorCode::NotImplemented,
              "InApplyFilter() not supported for RTreeIndex");
    return {};
}

template <typename T>
void
RTreeIndex<T>::InApplyCallback(size_t n,
                               const T* values,
                               const std::function<void(size_t)>& callback) {
    PanicInfo(ErrorCode::NotImplemented,
              "InApplyCallback() not supported for RTreeIndex");
}

template <typename T>
const TargetBitmap
RTreeIndex<T>::NotIn(size_t n, const T* values) {
    PanicInfo(ErrorCode::NotImplemented,
              "NotIn() not supported for RTreeIndex");
    return {};
}

template <typename T>
const TargetBitmap
RTreeIndex<T>::Range(T value, OpType op) {
    PanicInfo(ErrorCode::NotImplemented,
              "Range(value, op) not supported for RTreeIndex");
    return {};
}

template <typename T>
const TargetBitmap
RTreeIndex<T>::Range(T lower_bound_value,
                     bool lb_inclusive,
                     T upper_bound_value,
                     bool ub_inclusive) {
    PanicInfo(ErrorCode::NotImplemented,
              "Range(lower, upper) not supported for RTreeIndex");
    return {};
}

template <typename T>
void
RTreeIndex<T>::QueryCandidates(proto::plan::GISFunctionFilterExpr_GISOp op,
                               const std::string& query_geom_wkb,
                               std::vector<int64_t>& candidate_offsets) {
    PanicInfo(ErrorCode::NotImplemented,
              "QueryCandidates() not yet implemented for RTreeIndex");
}

template <typename T>
const TargetBitmap
RTreeIndex<T>::Query(const DatasetPtr& dataset) {
    // Empty implementation – to be filled by GIS spatial query in the future
    return {};
}

// ------------------------------------------------------------------
// BuildWithRawDataForUT – real implementation for unit-test scenarios
// ------------------------------------------------------------------

template <typename T>
void
RTreeIndex<T>::BuildWithRawDataForUT(size_t n,
                                     const void* values,
                                     const Config& config) {
    // In UT we directly receive an array of std::string (WKB) with length n.
    const std::string* wkb_array = reinterpret_cast<const std::string*>(values);

    // Guard: n should represent number of strings not raw bytes
    AssertInfo(n > 0, "BuildWithRawDataForUT expects element count > 0");
    LOG_WARN("BuildWithRawDataForUT:{}", n);
    this->InitForBuildIndex();

    int64_t offset = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto& wkb = wkb_array[i];
        const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(wkb.data());
        this->wrapper_->add_geometry(data_ptr, wkb.size(), offset++);
    }
    this->finish();
    LOG_WARN("BuildWithRawDataForUT finish");
    this->total_num_rows_ = offset;
    LOG_WARN("BuildWithRawDataForUT total_num_rows_:{}", this->total_num_rows_);
    this->is_built_ = true;
}

// Explicit template instantiation for std::string as we only support string field for now.
template class RTreeIndex<std::string>;

}  // namespace milvus::index