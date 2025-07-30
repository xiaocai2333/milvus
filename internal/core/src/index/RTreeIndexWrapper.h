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

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "ogr_geometry.h"
#include "spatialindex/SpatialIndex.h"
#include "common/Types.h"
#include "pb/plan.pb.h"
#include <folly/SharedMutex.h>

// Forward declaration to avoid pulling heavy field data headers here
namespace milvus {
class FieldDataBase;
}

namespace milvus::index {

/**
 * @brief Wrapper class for libspatialindex R-Tree functionality
 * 
 * This class provides a simplified interface to libspatialindex library,
 * handling the creation, management, and querying of R-Tree spatial indexes
 * for geometric data in Milvus.
 */
class RTreeIndexWrapper {
 public:
    /**
     * @brief Constructor for RTreeIndexWrapper
     * @param path Path for storing index files
     * @param is_build_mode Whether this is for building new index or loading existing one
     */
    explicit RTreeIndexWrapper(std::string& path, bool is_build_mode);

    /**
     * @brief Destructor
     */
    ~RTreeIndexWrapper();

    /**
     * @brief Add a geometry to the index
     * @param wkb_data Pointer to WKB binary data
     * @param len Length of WKB data
     * @param row_offset Row offset (used as identifier)
     */
    void
    add_geometry(const uint8_t* wkb_data, size_t len, int64_t row_offset);

    /**
     * @brief Bulk load geometries from field data (WKB strings) into a new R-Tree.
     *        This API will create the R-Tree via createAndBulkLoadNewRTree internally.
     * @param field_datas Vector of field data blocks containing WKB strings
     * @param nullable Whether the field allows nulls (null rows are skipped but offset still advances)
     */
    void
    bulk_load_from_field_data(
        const std::vector<std::shared_ptr<::milvus::FieldDataBase>>&
            field_datas,
        bool nullable);

    /**
     * @brief Finish building the index and flush to disk
     */
    void
    finish();

    /**
     * @brief Load existing index from disk
     */
    void
    load();

    /**
     * @brief Query candidates based on spatial operation
     * @param op Spatial operation type
     * @param query_geom Query geometry
     * @param candidate_offsets Output vector of candidate row offsets
     */
    void
    query_candidates(proto::plan::GISFunctionFilterExpr_GISOp op,
                     const OGRGeometry& query_geom,
                     std::vector<int64_t>& candidate_offsets);

    /**
     * @brief Get the total number of geometries in the index
     * @return Number of geometries
     */
    int64_t
    count() const;

    void
    set_rtree_variant(const std::string& variant_str);

    void
    set_fill_factor(double fill_factor);

    void
    set_index_capacity(uint32_t index_capacity);

    void
    set_leaf_capacity(uint32_t leaf_capacity);

 private:
    /**
     * @brief Get bounding box from OGR geometry
     * @param geom Input geometry
     * @param minX Output minimum X coordinate
     * @param minY Output minimum Y coordinate
     * @param maxX Output maximum X coordinate
     * @param maxY Output maximum Y coordinate
     */
    void
    get_bounding_box(const OGRGeometry* geom,
                     double& minX,
                     double& minY,
                     double& maxX,
                     double& maxY);

 private:
    std::shared_ptr<SpatialIndex::IStorageManager> storage_manager_;
    std::shared_ptr<SpatialIndex::ISpatialIndex> rtree_;
    std::string index_path_;
    bool is_build_mode_;

    // Flag to guard against repeated invocations which could otherwise attempt to release resources multiple times (e.g. BuildWithRawDataForUT() calls finish(), and Upload() may call it again).
    bool finished_ = false;
    SpatialIndex::id_type index_id_ = 0;  // persisted to meta for reliable load

    // R-Tree parameters
    double fill_factor_ = 0.8;
    uint32_t index_capacity_ = 50;
    uint32_t leaf_capacity_ = 50;
    uint32_t dimension_ = 2;
    SpatialIndex::RTree::RTreeVariant rtree_variant_ =
        SpatialIndex::RTree::RV_RSTAR;

    // Thread safety: protects rtree_ and related operations
    mutable folly::SharedMutexWritePriority rtree_mutex_;
};

}  // namespace milvus::index