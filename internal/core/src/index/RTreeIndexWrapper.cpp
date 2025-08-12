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

#include "RTreeIndexWrapper.h"
#include "common/EasyAssert.h"
#include "log/Log.h"
#include "pb/plan.pb.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include "common/FieldDataInterface.h"

namespace milvus::index {

// Custom visitor for collecting query results
class GeometryVisitor : public SpatialIndex::IVisitor {
 public:
    explicit GeometryVisitor(std::vector<int64_t>& results)
        : results_(results) {
    }

    virtual ~GeometryVisitor() = default;

    void
    visitNode(const SpatialIndex::INode& n) override {
        // Not needed for our use case
    }

    void
    visitData(const SpatialIndex::IData& d) override {
        // Store the identifier (row offset) in results
        results_.push_back(static_cast<int64_t>(d.getIdentifier()));
    }

    void
    visitData(std::vector<const SpatialIndex::IData*>& v) override {
        for (const auto* data : v) {
            results_.push_back(static_cast<int64_t>(data->getIdentifier()));
        }
    }

 private:
    std::vector<int64_t>& results_;
};

RTreeIndexWrapper::RTreeIndexWrapper(std::string& path, bool is_build_mode)
    : index_path_(path), is_build_mode_(is_build_mode) {
    if (is_build_mode_) {
        // Create directory if it doesn't exist
        std::filesystem::path dir_path =
            std::filesystem::path(path).parent_path();
        if (!dir_path.empty()) {
            std::filesystem::create_directories(dir_path);
        }

        // Create disk storage manager for building
        storage_manager_ = std::shared_ptr<SpatialIndex::IStorageManager>(
            SpatialIndex::StorageManager::createNewDiskStorageManager(path,
                                                                      4096));
    }
}

RTreeIndexWrapper::~RTreeIndexWrapper() = default;

void
RTreeIndexWrapper::add_geometry(const uint8_t* wkb_data,
                                size_t len,
                                int64_t row_offset) {
    AssertInfo(is_build_mode_, "Cannot add geometry in load mode");
    // Lazily create the R-Tree for dynamic insertion if not present yet
    if (rtree_ == nullptr) {
        SpatialIndex::id_type index_id;
        rtree_ = std::shared_ptr<SpatialIndex::ISpatialIndex>(
            SpatialIndex::RTree::createNewRTree(*storage_manager_,
                                                fill_factor_,
                                                index_capacity_,
                                                leaf_capacity_,
                                                dimension_,
                                                rtree_variant_,
                                                index_id));
        index_id_ = index_id;
        LOG_WARN("create rtree index for dynamic insertion");
    }

    // Parse WKB data to OGR geometry
    OGRGeometry* geom = nullptr;
    OGRErr err =
        OGRGeometryFactory::createFromWkb(wkb_data, nullptr, &geom, len);

    if (err != OGRERR_NONE || geom == nullptr) {
        LOG_ERROR("Failed to parse WKB data for row {}", row_offset);
        return;
    }

    // Get bounding box
    double minX, minY, maxX, maxY;
    get_bounding_box(geom, minX, minY, maxX, maxY);

    // Create region for the bounding box
    double low[2] = {minX, minY};
    double high[2] = {maxX, maxY};
    SpatialIndex::Region region(low, high, 2);

    // Insert into R-Tree with row_offset as identifier
    rtree_->insertData(
        0, nullptr, region, static_cast<SpatialIndex::id_type>(row_offset));

    // Clean up
    OGRGeometryFactory::destroyGeometry(geom);
}

// Internal IDataStream implementation over FieldDataBase (WKB string rows)
namespace {
class BulkLoadDataStream : public SpatialIndex::IDataStream {
 public:
    BulkLoadDataStream(
        const std::vector<std::shared_ptr<::milvus::FieldDataBase>>&
            field_datas,
        bool nullable)
        : field_datas_(field_datas), nullable_param_(nullable) {
        // Compute a cheap upper bound for stream size: sum of row counts
        total_rows_ = 0;
        for (const auto& fd : field_datas_) {
            total_rows_ += static_cast<size_t>(fd->get_num_rows());
        }
        rewind();
    }

    ~BulkLoadDataStream() override = default;

    bool
    hasNext() override {
        return absolute_offset_ < static_cast<int64_t>(total_rows_);
    }

    uint32_t
    size() override {
        // Return upper bound; actual yielded items may be fewer due to
        // null rows or invalid WKB filtered in getNext().
        return static_cast<uint32_t>(total_rows_);
    }

    void
    rewind() override {
        batch_index_ = 0;
        row_in_batch_ = 0;
        absolute_offset_ = 0;
    }

    SpatialIndex::IData*
    getNext() override {
        while (batch_index_ < field_datas_.size()) {
            const auto& fd = field_datas_[batch_index_];
            auto n = fd->get_num_rows();
            if (row_in_batch_ >= n) {
                ++batch_index_;
                row_in_batch_ = 0;
                continue;
            }

            int64_t current_row_in_batch = row_in_batch_;
            int64_t current_abs = absolute_offset_;
            // advance offsets for next call regardless of validity
            ++row_in_batch_;
            ++absolute_offset_;

            const bool is_nullable_effective =
                nullable_param_ || fd->IsNullable();
            if (is_nullable_effective && !fd->is_valid(current_row_in_batch)) {
                null_rows_++;
                continue;
            }

            const auto* wkb_str = static_cast<const std::string*>(
                fd->RawValue(current_row_in_batch));
            if (wkb_str == nullptr || wkb_str->empty()) {
                continue;
            }

            // Parse WKB using OGR to get envelope
            OGRGeometry* geom = nullptr;
            OGRErr err = OGRGeometryFactory::createFromWkb(
                reinterpret_cast<const uint8_t*>(wkb_str->data()),
                nullptr,
                &geom,
                wkb_str->size());
            if (err != OGRERR_NONE || geom == nullptr) {
                LOG_WARN(
                    "BulkLoadDataStream: failed to parse WKB at abs {} (batch "
                    "{}, row {})",
                    current_abs,
                    batch_index_,
                    current_row_in_batch);
                continue;
            }

            OGREnvelope env;
            geom->getEnvelope(&env);
            OGRGeometryFactory::destroyGeometry(geom);

            double low[2] = {env.MinX, env.MinY};
            double high[2] = {env.MaxX, env.MaxY};
            SpatialIndex::Region region(low, high, 2);

            return new SpatialIndex::RTree::Data(
                0,
                nullptr,
                region,
                static_cast<SpatialIndex::id_type>(current_abs));
        }
        return nullptr;
    }

    int64_t
    null_rows() const {
        return null_rows_;
    }

 private:
    const std::vector<std::shared_ptr<::milvus::FieldDataBase>>& field_datas_;
    bool nullable_param_ = false;
    size_t total_rows_ = 0;
    size_t batch_index_ = 0;
    int64_t row_in_batch_ = 0;
    int64_t absolute_offset_ = 0;
    int64_t null_rows_ = 0;
};
}  // anonymous namespace

void
RTreeIndexWrapper::bulk_load_from_field_data(
    const std::vector<std::shared_ptr<::milvus::FieldDataBase>>& field_datas,
    bool nullable) {
    AssertInfo(is_build_mode_, "Cannot bulk load in load mode");
    AssertInfo(storage_manager_ != nullptr, "Storage manager is null");
    AssertInfo(rtree_ == nullptr,
               "R-Tree already initialized; bulk load requires a fresh tree");

    BulkLoadDataStream stream(field_datas, nullable);
    SpatialIndex::id_type index_id;
    try {
        rtree_ = std::shared_ptr<SpatialIndex::ISpatialIndex>(
            SpatialIndex::RTree::createAndBulkLoadNewRTree(
                SpatialIndex::RTree::BLM_STR,
                stream,
                *storage_manager_,
                fill_factor_,
                index_capacity_,
                leaf_capacity_,
                dimension_,
                rtree_variant_,
                index_id));
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to bulk load R-Tree: {}", e.what());
    }

    index_id_ = index_id;
    null_rows_ = stream.null_rows();
    LOG_INFO("R-Tree bulk load completed with {} entries",
             rtree_ ? "some" : "none");
}

void
RTreeIndexWrapper::finish() {
    // Guard against repeated invocations which could otherwise attempt to
    // release resources multiple times (e.g. BuildWithRawDataForUT() calls
    // finish(), and Upload() may call it again).
    if (finished_) {
        LOG_DEBUG("RTreeIndexWrapper::finish() called more than once, skip.");
        return;
    }

    AssertInfo(is_build_mode_, "Cannot finish in load mode");

    // If rtree_ is already reset, we have nothing left to do. Mark finished
    // and return.
    if (rtree_ == nullptr) {
        LOG_DEBUG(
            "RTreeIndexWrapper::finish() called with null rtree_, likely "
            "already finished.");
        finished_ = true;
        return;
    }

    // Explicitly flush the index header & buffers to disk to guarantee
    // consistency before releasing resources.
    rtree_->flush();

    // NOTE: rtree_ internally holds a pointer to the storage manager. We must
    // make sure rtree_ is destroyed BEFORE the storage manager.

    // 1. Release rtree_ first so its destructor can safely write the header
    //    using a still-valid storage_manager_.
    rtree_.reset();

    // 2. Now it is safe to release the storage manager.
    storage_manager_.reset();

    // 3. Write meta file with index parameters for reliable loading.
    try {
        nlohmann::json meta;
        meta["index_id"] = index_id_;
        meta["variant"] = static_cast<int>(rtree_variant_);
        meta["fill_factor"] = fill_factor_;
        meta["index_capacity"] = index_capacity_;
        meta["leaf_capacity"] = leaf_capacity_;
        meta["dimension"] = dimension_;
        meta["null_rows"] = null_rows_;

        std::ofstream ofs(index_path_ + ".meta.json", std::ios::trunc);
        ofs << meta.dump();
        ofs.close();
        LOG_INFO("R-Tree meta written: {}.meta.json", index_path_);
    } catch (const std::exception& e) {
        LOG_WARN("Failed to write R-Tree meta json: {}", e.what());
    }

    finished_ = true;

    LOG_INFO("R-Tree index finished building and saved to {}", index_path_);
}

void
RTreeIndexWrapper::load() {
    AssertInfo(!is_build_mode_, "Cannot load in build mode");

    try {
        // Load storage manager
        storage_manager_ = std::shared_ptr<SpatialIndex::IStorageManager>(
            SpatialIndex::StorageManager::loadDiskStorageManager(index_path_));

        // Determine index id from meta json if available
        SpatialIndex::id_type idx_id_to_load = 0;
        try {
            std::ifstream ifs(index_path_ + ".meta.json");
            if (ifs.good()) {
                auto meta = nlohmann::json::parse(ifs);
                if (meta.contains("index_id")) {
                    idx_id_to_load =
                        meta["index_id"].get<SpatialIndex::id_type>();
                }
                if (meta.contains("null_rows")) {
                    null_rows_ = meta["null_rows"].get<int64_t>();
                }
            }
        } catch (const std::exception& e) {
            LOG_WARN("Failed to read meta json, fallback to default id 0: {}",
                     e.what());
        }

        // Load R-Tree index with the resolved id
        rtree_ = std::shared_ptr<SpatialIndex::ISpatialIndex>(
            SpatialIndex::RTree::loadRTree(*storage_manager_, idx_id_to_load));

        LOG_INFO("R-Tree index loaded from {}", index_path_);
    } catch (const std::exception& e) {
        PanicInfo(ErrorCode::UnexpectedError,
                  fmt::format("Failed to load R-Tree index from {}: {}",
                              index_path_,
                              e.what()));
    }
}

void
RTreeIndexWrapper::query_candidates(proto::plan::GISFunctionFilterExpr_GISOp op,
                                    const OGRGeometry& query_geom,
                                    std::vector<int64_t>& candidate_offsets) {
    AssertInfo(rtree_ != nullptr, "R-Tree index not initialized");

    candidate_offsets.clear();

    // Get bounding box of query geometry
    double minX, minY, maxX, maxY;
    get_bounding_box(&query_geom, minX, minY, maxX, maxY);

    // Create query region
    double low[2] = {minX, minY};
    double high[2] = {maxX, maxY};
    SpatialIndex::Region query_region(low, high, 2);

    // Create visitor for collecting results
    GeometryVisitor visitor(candidate_offsets);

    // Perform query based on operation type
    switch (op) {
        case proto::plan::GISFunctionFilterExpr_GISOp_Contains:
            rtree_->containsWhatQuery(query_region, visitor);
            break;
        default:
            // For all GIS operations, we use intersection query as coarse filtering
            // The exact geometric relationship will be checked in the refinement phase
            rtree_->intersectsWithQuery(query_region, visitor);
            break;
    }

    LOG_DEBUG("R-Tree query returned {} candidates for operation {}",
              candidate_offsets.size(),
              static_cast<int>(op));
}

void
RTreeIndexWrapper::get_bounding_box(const OGRGeometry* geom,
                                    double& minX,
                                    double& minY,
                                    double& maxX,
                                    double& maxY) {
    AssertInfo(geom != nullptr, "Geometry is null");

    OGREnvelope env;
    geom->getEnvelope(&env);

    minX = env.MinX;
    minY = env.MinY;
    maxX = env.MaxX;
    maxY = env.MaxY;
}

int64_t
RTreeIndexWrapper::count() const {
    if (rtree_ == nullptr) {
        return 0;
    }

    // For R-Tree, we need to count the number of data entries
    // This is a simplified implementation - in practice, you might want to
    // maintain a separate counter during building
    SpatialIndex::IStatistics* stats = nullptr;
    rtree_->getStatistics(&stats);
    if (stats != nullptr) {
        int64_t count = stats->getNumberOfData();
        delete stats;
        return count;
    }
    return 0;
}

void
RTreeIndexWrapper::set_rtree_variant(const std::string& variant_str) {
    if (variant_str == "RSTAR") {
        rtree_variant_ = SpatialIndex::RTree::RV_RSTAR;
    } else if (variant_str == "QUADRATIC") {
        LOG_WARN("QUADRATIC variant is not supported, using RSTAR instead");
        rtree_variant_ = SpatialIndex::RTree::RV_RSTAR;
    } else if (variant_str == "LINEAR") {
        rtree_variant_ = SpatialIndex::RTree::RV_LINEAR;
    } else {
        PanicInfo(ErrorCode::UnexpectedError,
                  fmt::format("Invalid R-Tree variant: {}", variant_str));
    }
}

void
RTreeIndexWrapper::set_fill_factor(double fill_factor) {
    fill_factor_ = fill_factor;
}

void
RTreeIndexWrapper::set_index_capacity(uint32_t index_capacity) {
    index_capacity_ = index_capacity;
}

void
RTreeIndexWrapper::set_leaf_capacity(uint32_t leaf_capacity) {
    leaf_capacity_ = leaf_capacity;
}
}  // namespace milvus::index