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

        // Create R-Tree index
        SpatialIndex::id_type index_id;
        rtree_ = std::shared_ptr<SpatialIndex::ISpatialIndex>(
            SpatialIndex::RTree::createNewRTree(*storage_manager_,
                                                fill_factor_,
                                                index_capacity_,
                                                leaf_capacity_,
                                                dimension_,
                                                rtree_variant_,
                                                index_id));
    }
}

RTreeIndexWrapper::~RTreeIndexWrapper() = default;

void
RTreeIndexWrapper::add_geometry(const uint8_t* wkb_data,
                                size_t len,
                                int64_t row_offset) {
    AssertInfo(is_build_mode_, "Cannot add geometry in load mode");
    AssertInfo(rtree_ != nullptr, "R-Tree index not initialized");

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

void
RTreeIndexWrapper::finish() {
    AssertInfo(is_build_mode_, "Cannot finish in load mode");
    AssertInfo(rtree_ != nullptr, "R-Tree index not initialized");

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

    LOG_INFO("R-Tree index finished building and saved to {}", index_path_);
}

void
RTreeIndexWrapper::load() {
    AssertInfo(!is_build_mode_, "Cannot load in build mode");

    try {
        // Load storage manager
        storage_manager_ = std::shared_ptr<SpatialIndex::IStorageManager>(
            SpatialIndex::StorageManager::loadDiskStorageManager(index_path_));

        // Load R-Tree index
        rtree_ = std::shared_ptr<SpatialIndex::ISpatialIndex>(
            SpatialIndex::RTree::loadRTree(*storage_manager_,
                                           1));  // 1 is the index ID

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

}  // namespace milvus::index