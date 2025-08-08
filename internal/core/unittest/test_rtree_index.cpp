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

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>
#include <string>

#include "index/RTreeIndex.h"
#include "storage/Util.h"
#include "storage/FileManager.h"
#include "common/Types.h"
#include "test_utils/TmpPath.h"
#include "pb/schema.pb.h"
#include "pb/plan.pb.h"
#include "common/Geometry.h"
#include "common/EasyAssert.h"
#include "storage/InsertData.h"
#include "storage/PayloadReader.h"
#include "storage/DiskFileManagerImpl.h"
#include "common/FieldData.h"
#include <boost/algorithm/string/predicate.hpp>
#include <fstream>

// Helper: create simple POINT(x,y) WKB (little-endian)
static std::string
CreatePointWKB(double x, double y) {
    std::vector<uint8_t> wkb;
    // Byte order – little endian (1)
    wkb.push_back(0x01);
    // Geometry type – Point (1) – 32-bit little endian
    uint32_t geom_type = 1;
    uint8_t* type_bytes = reinterpret_cast<uint8_t*>(&geom_type);
    wkb.insert(wkb.end(), type_bytes, type_bytes + sizeof(uint32_t));
    // X coordinate
    uint8_t* x_bytes = reinterpret_cast<uint8_t*>(&x);
    wkb.insert(wkb.end(), x_bytes, x_bytes + sizeof(double));
    // Y coordinate
    uint8_t* y_bytes = reinterpret_cast<uint8_t*>(&y);
    wkb.insert(wkb.end(), y_bytes, y_bytes + sizeof(double));
    return std::string(reinterpret_cast<const char*>(wkb.data()), wkb.size());
}

// Helper: create simple WKB from WKT
static std::string
CreateWkbFromWkt(const std::string& wkt) {
    return milvus::Geometry(wkt.c_str()).to_wkb_string();
}

// Helper: write an InsertData parquet file to "remote" storage managed by chunk_manager_
static std::string
WriteGeometryInsertFile(const milvus::storage::ChunkManagerPtr& cm,
                        const milvus::storage::FieldDataMeta& field_meta,
                        const std::string& remote_path,
                        const std::vector<std::string>& wkbs,
                        bool nullable = false,
                        const uint8_t* valid_bitmap = nullptr) {
    auto field_data = milvus::storage::CreateFieldData(
        milvus::storage::DataType::GEOMETRY, nullable);
    if (nullable && valid_bitmap != nullptr) {
        field_data->FillFieldData(wkbs.data(), valid_bitmap, wkbs.size());
    } else {
        field_data->FillFieldData(wkbs.data(), wkbs.size());
    }
    auto payload_reader =
        std::make_shared<milvus::storage::PayloadReader>(field_data);
    milvus::storage::InsertData insert_data(payload_reader);
    insert_data.SetFieldDataMeta(field_meta);
    insert_data.SetTimestamps(0, 100);

    auto bytes = insert_data.Serialize(milvus::storage::StorageType::Remote);
    std::vector<uint8_t> buf(bytes.begin(), bytes.end());
    cm->Write(remote_path, buf.data(), buf.size());
    return remote_path;
}

class RTreeIndexTest : public ::testing::Test {
 protected:
    void
    SetUp() override {
        temp_path_ = milvus::test::TmpPath{};
        // create storage config that writes to temp dir
        storage_config_.storage_type = "local";
        storage_config_.root_path = temp_path_.get().string();
        chunk_manager_ = milvus::storage::CreateChunkManager(storage_config_);

        // prepare field & index meta – minimal info for DiskFileManagerImpl
        field_meta_ = milvus::storage::FieldDataMeta{1, 1, 1, 100};
        index_meta_ = milvus::storage::IndexMeta{.segment_id = 1,
                                                 .field_id = 100,
                                                 .build_id = 1,
                                                 .index_version = 1};
    }

    void
    TearDown() override {
        // clean chunk manager files if any (TmpPath destructor will also remove)
    }

    milvus::storage::StorageConfig storage_config_;
    milvus::storage::ChunkManagerPtr chunk_manager_;
    milvus::storage::FieldDataMeta field_meta_;
    milvus::storage::IndexMeta index_meta_;
    milvus::test::TmpPath temp_path_;
};

TEST_F(RTreeIndexTest, Build_Upload_Load) {
    // ---------- Build via BuildWithRawDataForUT ----------
    milvus::storage::FileManagerContext ctx_build(
        field_meta_, index_meta_, chunk_manager_);
    milvus::index::RTreeIndex<std::string> rtree_build(ctx_build);

    std::vector<std::string> wkbs = {CreatePointWKB(1.0, 1.0),
                                     CreatePointWKB(2.0, 2.0)};
    rtree_build.BuildWithRawDataForUT(wkbs.size(), wkbs.data());

    ASSERT_EQ(rtree_build.Count(), 2);

    // ---------- Upload ----------
    auto stats = rtree_build.Upload({});
    ASSERT_NE(stats, nullptr);
    ASSERT_GT(stats->GetIndexFiles().size(), 0);

    // ---------- Load back ----------
    milvus::storage::FileManagerContext ctx_load(
        field_meta_, index_meta_, chunk_manager_);
    ctx_load.set_for_loading_index(true);
    milvus::index::RTreeIndex<std::string> rtree_load(ctx_load);

    nlohmann::json cfg;
    cfg["index_files"] = stats->GetIndexFiles();

    milvus::tracer::TraceContext trace_ctx;  // empty context
    rtree_load.Load(trace_ctx, cfg);

    ASSERT_EQ(rtree_load.Count(), 2);
}

TEST_F(RTreeIndexTest, Load_WithFileNamesOnly) {
    // Build & upload first
    milvus::storage::FileManagerContext ctx_build(
        field_meta_, index_meta_, chunk_manager_);
    milvus::index::RTreeIndex<std::string> rtree_build(ctx_build);

    std::vector<std::string> wkbs2 = {CreatePointWKB(10.0, 10.0),
                                      CreatePointWKB(20.0, 20.0)};
    rtree_build.BuildWithRawDataForUT(wkbs2.size(), wkbs2.data());

    auto stats = rtree_build.Upload({});

    // gather only filenames (strip parent path)
    std::vector<std::string> filenames;
    for (const auto& path : stats->GetIndexFiles()) {
        filenames.emplace_back(
            boost::filesystem::path(path).filename().string());
        // make sure file exists in remote storage
        ASSERT_TRUE(chunk_manager_->Exist(path));
        ASSERT_GT(chunk_manager_->Size(path), 0);
    }

    // Load using filename only list
    milvus::storage::FileManagerContext ctx_load(
        field_meta_, index_meta_, chunk_manager_);
    ctx_load.set_for_loading_index(true);
    milvus::index::RTreeIndex<std::string> rtree_load(ctx_load);

    nlohmann::json cfg;
    cfg["index_files"] = filenames;  // no directory info

    milvus::tracer::TraceContext trace_ctx;
    rtree_load.Load(trace_ctx, cfg);

    ASSERT_EQ(rtree_load.Count(), 2);
}

TEST_F(RTreeIndexTest, Build_EmptyInput_ShouldThrow) {
    milvus::storage::FileManagerContext ctx(
        field_meta_, index_meta_, chunk_manager_);
    milvus::index::RTreeIndex<std::string> rtree(ctx);

    std::vector<std::string> empty;
    EXPECT_THROW(rtree.BuildWithRawDataForUT(0, empty.data()),
                 milvus::SegcoreError);
}

TEST_F(RTreeIndexTest, Build_WithInvalidWKB_Upload_Load) {
    milvus::storage::FileManagerContext ctx(
        field_meta_, index_meta_, chunk_manager_);
    milvus::index::RTreeIndex<std::string> rtree(ctx);

    std::string bad = CreatePointWKB(0.0, 0.0);
    bad.resize(bad.size() / 2);  // truncate to make invalid

    std::vector<std::string> wkbs = {
        CreateWkbFromWkt("POINT(1 1)"), bad, CreateWkbFromWkt("POINT(2 2)")};
    rtree.BuildWithRawDataForUT(wkbs.size(), wkbs.data());

    // Upload and then load back to let loader compute count from wrapper
    auto stats = rtree.Upload({});

    milvus::storage::FileManagerContext ctx_load(
        field_meta_, index_meta_, chunk_manager_);
    ctx_load.set_for_loading_index(true);
    milvus::index::RTreeIndex<std::string> rtree_load(ctx_load);

    nlohmann::json cfg;
    cfg["index_files"] = stats->GetIndexFiles();
    milvus::tracer::TraceContext trace_ctx;
    rtree_load.Load(trace_ctx, cfg);

    // Only 2 valid points should be present
    ASSERT_EQ(rtree_load.Count(), 2);
}

TEST_F(RTreeIndexTest, Build_VariousGeometries) {
    milvus::storage::FileManagerContext ctx(
        field_meta_, index_meta_, chunk_manager_);
    milvus::index::RTreeIndex<std::string> rtree(ctx);

    std::vector<std::string> wkbs = {
        CreateWkbFromWkt("POINT(-1.5 2.5)"),
        CreateWkbFromWkt("LINESTRING(0 0,1 1,2 3)"),
        CreateWkbFromWkt("POLYGON((0 0,2 0,2 2,0 2,0 0))"),
        CreateWkbFromWkt("POINT(1000000 -1000000)"),
        CreateWkbFromWkt("POINT(0 0)")};

    rtree.BuildWithRawDataForUT(wkbs.size(), wkbs.data());
    ASSERT_EQ(rtree.Count(), wkbs.size());

    auto stats = rtree.Upload({});
    ASSERT_FALSE(stats->GetIndexFiles().empty());

    milvus::storage::FileManagerContext ctx_load(
        field_meta_, index_meta_, chunk_manager_);
    ctx_load.set_for_loading_index(true);
    milvus::index::RTreeIndex<std::string> rtree_load(ctx_load);

    nlohmann::json cfg;
    cfg["index_files"] = stats->GetIndexFiles();
    milvus::tracer::TraceContext trace_ctx;
    rtree_load.Load(trace_ctx, cfg);
    ASSERT_EQ(rtree_load.Count(), wkbs.size());
}

TEST_F(RTreeIndexTest, Build_ConfigAndMetaJson) {
    // Prepare one insert file via storage pipeline
    std::vector<std::string> wkbs = {CreateWkbFromWkt("POINT(0 0)"),
                                     CreateWkbFromWkt("POINT(1 1)")};
    auto remote_file = (temp_path_.get() / "geom.parquet").string();
    WriteGeometryInsertFile(chunk_manager_, field_meta_, remote_file, wkbs);
    std::cout<<"JsonTest: test_rtree_index.cpp:260"<<std::endl;
    milvus::storage::FileManagerContext ctx(
        field_meta_, index_meta_, chunk_manager_);
    milvus::index::RTreeIndex<std::string> rtree(ctx);

    nlohmann::json build_cfg;
    build_cfg["insert_files"] = std::vector<std::string>{remote_file};
    build_cfg["fillFactor"] = 0.6;
    build_cfg["indexCapacity"] = 32;
    build_cfg["leafCapacity"] = 64;
    build_cfg["rv"] = "RSTAR";
    
    rtree.Build(build_cfg);
    auto stats = rtree.Upload({});

    // Cache remote index files locally
    milvus::storage::DiskFileManagerImpl diskfm(
        {field_meta_, index_meta_, chunk_manager_});
    auto index_files = stats->GetIndexFiles();
    diskfm.CacheIndexToDisk(index_files);
    auto local_paths = diskfm.GetLocalFilePaths();
    ASSERT_FALSE(local_paths.empty());
    std::cout<<"JsonTest: test_rtree_index.cpp:282"<<std::endl;
    // Determine base path like RTreeIndex::Load
    auto ends_with = [](const std::string& value, const std::string& suffix) {
        return value.size() >= suffix.size() &&
               value.compare(
                   value.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

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
    if (base_path.empty()) {
        for (const auto& p : local_paths) {
            if (ends_with(p, ".meta.json")) {
                base_path =
                    p.substr(0, p.size() - std::string(".meta.json").size());
                break;
            }
        }
    }
    if (base_path.empty()) {
        base_path = local_paths.front();
    }
    // Parse local meta json
    std::ifstream ifs(base_path + ".meta.json");
    ASSERT_TRUE(ifs.good());
    nlohmann::json meta = nlohmann::json::parse(ifs);
    ASSERT_EQ(meta["fill_factor"], 0.6);
    ASSERT_EQ(meta["index_capacity"], 32);
    ASSERT_EQ(meta["leaf_capacity"], 64);
    ASSERT_EQ(meta["dimension"], 2);
}

TEST_F(RTreeIndexTest, Build_InvalidVariant_ShouldThrow) {
    // Prepare insert
    std::vector<std::string> wkbs = {CreateWkbFromWkt("POINT(0 0)")};
    auto remote_file = (temp_path_.get() / "geom2.parquet").string();
    WriteGeometryInsertFile(chunk_manager_, field_meta_, remote_file, wkbs);

    milvus::storage::FileManagerContext ctx(
        field_meta_, index_meta_, chunk_manager_);
    milvus::index::RTreeIndex<std::string> rtree(ctx);

    nlohmann::json build_cfg;
    build_cfg["insert_files"] = std::vector<std::string>{remote_file};
    build_cfg["rv"] = "FOO";  // invalid
    EXPECT_THROW(rtree.Build(build_cfg), milvus::SegcoreError);
}

TEST_F(RTreeIndexTest, Load_OnlyIdx_OnlyDat) {
    // Build and upload
    milvus::storage::FileManagerContext ctx(
        field_meta_, index_meta_, chunk_manager_);
    milvus::index::RTreeIndex<std::string> rtree(ctx);
    std::vector<std::string> wkbs = {CreatePointWKB(3.0, 3.0),
                                     CreatePointWKB(4.0, 4.0)};
    rtree.BuildWithRawDataForUT(wkbs.size(), wkbs.data());
    auto stats = rtree.Upload({});

    std::vector<std::string> only_idx, only_dat;
    for (const auto& p : stats->GetIndexFiles()) {
        if (boost::algorithm::ends_with(p, ".idx_0"))
            only_idx.push_back(p);
        if (boost::algorithm::ends_with(p, ".dat_0"))
            only_dat.push_back(p);
    }
    ASSERT_FALSE(only_idx.empty());
    ASSERT_FALSE(only_dat.empty());

    // Load with only idx should fail
    milvus::storage::FileManagerContext ctx_load1(
        field_meta_, index_meta_, chunk_manager_);
    ctx_load1.set_for_loading_index(true);
    milvus::index::RTreeIndex<std::string> rtree_load1(ctx_load1);
    nlohmann::json cfg1;
    cfg1["index_files"] = only_idx;
    milvus::tracer::TraceContext trace_ctx;
    EXPECT_ANY_THROW(rtree_load1.Load(trace_ctx, cfg1));

    // Load with only dat should fail
    milvus::storage::FileManagerContext ctx_load2(
        field_meta_, index_meta_, chunk_manager_);
    ctx_load2.set_for_loading_index(true);
    milvus::index::RTreeIndex<std::string> rtree_load2(ctx_load2);
    nlohmann::json cfg2;
    cfg2["index_files"] = only_dat;
    EXPECT_ANY_THROW(rtree_load2.Load(trace_ctx, cfg2));
}

TEST_F(RTreeIndexTest, Load_OnlyMeta_ShouldThrow) {
    // Build and upload
    milvus::storage::FileManagerContext ctx(
        field_meta_, index_meta_, chunk_manager_);
    milvus::index::RTreeIndex<std::string> rtree(ctx);
    std::vector<std::string> wkbs = {CreatePointWKB(5.0, 5.0)};
    rtree.BuildWithRawDataForUT(wkbs.size(), wkbs.data());
    auto stats = rtree.Upload({});

    std::vector<std::string> only_meta;
    for (const auto& p : stats->GetIndexFiles()) {
        if (boost::algorithm::ends_with(p, ".meta.json_0"))
            only_meta.push_back(p);
    }
    ASSERT_FALSE(only_meta.empty());

    milvus::storage::FileManagerContext ctx_load(
        field_meta_, index_meta_, chunk_manager_);
    ctx_load.set_for_loading_index(true);
    milvus::index::RTreeIndex<std::string> rtree_load(ctx_load);
    nlohmann::json cfg;
    cfg["index_files"] = only_meta;
    milvus::tracer::TraceContext trace_ctx;
    EXPECT_ANY_THROW(rtree_load.Load(trace_ctx, cfg));
}

TEST_F(RTreeIndexTest, Load_MixedFileNamesAndPaths) {
    // Build and upload
    milvus::storage::FileManagerContext ctx(
        field_meta_, index_meta_, chunk_manager_);
    milvus::index::RTreeIndex<std::string> rtree(ctx);
    std::vector<std::string> wkbs = {CreatePointWKB(6.0, 6.0),
                                     CreatePointWKB(7.0, 7.0)};
    rtree.BuildWithRawDataForUT(wkbs.size(), wkbs.data());
    auto stats = rtree.Upload({});

    // Use full list, but replace one with filename-only
    auto mixed = stats->GetIndexFiles();
    ASSERT_FALSE(mixed.empty());
    mixed[0] = boost::filesystem::path(mixed[0]).filename().string();

    milvus::storage::FileManagerContext ctx_load(
        field_meta_, index_meta_, chunk_manager_);
    ctx_load.set_for_loading_index(true);
    milvus::index::RTreeIndex<std::string> rtree_load(ctx_load);

    nlohmann::json cfg;
    cfg["index_files"] = mixed;
    milvus::tracer::TraceContext trace_ctx;
    rtree_load.Load(trace_ctx, cfg);
    ASSERT_EQ(rtree_load.Count(), wkbs.size());
}

TEST_F(RTreeIndexTest, Load_NonexistentRemote_ShouldThrow) {
    milvus::storage::FileManagerContext ctx_load(
        field_meta_, index_meta_, chunk_manager_);
    ctx_load.set_for_loading_index(true);
    milvus::index::RTreeIndex<std::string> rtree_load(ctx_load);

    // nonexist file
    nlohmann::json cfg;
    cfg["index_files"] = std::vector<std::string>{
        (temp_path_.get() / "does_not_exist.idx_0").string()};
    milvus::tracer::TraceContext trace_ctx;
    EXPECT_THROW(rtree_load.Load(trace_ctx, cfg), milvus::SegcoreError);
}

TEST_F(RTreeIndexTest, API_NotImplemented_Throws) {
    milvus::storage::FileManagerContext ctx(
        field_meta_, index_meta_, chunk_manager_);
    milvus::index::RTreeIndex<std::string> rtree(ctx);

    EXPECT_THROW(rtree.Serialize({}), milvus::SegcoreError);

    milvus::BinarySet bs;
    EXPECT_THROW(rtree.Load(bs, {}), milvus::SegcoreError);

    std::string dummy;
    EXPECT_THROW(rtree.In(1, &dummy), milvus::SegcoreError);
    EXPECT_THROW(rtree.IsNull(), milvus::SegcoreError);
    EXPECT_THROW(rtree.IsNotNull(), milvus::SegcoreError);
    EXPECT_THROW(rtree.InApplyFilter(1, &dummy, [](size_t) { return true; }),
                 milvus::SegcoreError);
    EXPECT_THROW(rtree.InApplyCallback(1, &dummy, [](size_t) {}),
                 milvus::SegcoreError);
    EXPECT_THROW(rtree.NotIn(1, &dummy), milvus::SegcoreError);
    EXPECT_THROW(rtree.Range(dummy, milvus::OpType::GreaterEqual),
                 milvus::SegcoreError);
    EXPECT_THROW(rtree.Range(dummy, true, dummy, false), milvus::SegcoreError);
    std::vector<int64_t> candidates;
    EXPECT_THROW(
        rtree.QueryCandidates(
            ::milvus::proto::plan::GISFunctionFilterExpr_GISOp_Contains,
            CreateWkbFromWkt("POINT(0 0)"),
            candidates),
        milvus::SegcoreError);
}

TEST_F(RTreeIndexTest, Build_EndToEnd_FromInsertFiles) {
    // prepare remote file via InsertData serialization
    std::vector<std::string> wkbs = {CreateWkbFromWkt("POINT(0 0)"),
                                     CreateWkbFromWkt("POINT(2 2)")};
    auto remote_file = (temp_path_.get() / "geom3.parquet").string();
    WriteGeometryInsertFile(chunk_manager_, field_meta_, remote_file, wkbs);

    milvus::storage::FileManagerContext ctx(
        field_meta_, index_meta_, chunk_manager_);
    milvus::index::RTreeIndex<std::string> rtree(ctx);

    nlohmann::json build_cfg;
    build_cfg["insert_files"] = std::vector<std::string>{remote_file};
    rtree.Build(build_cfg);
    ASSERT_EQ(rtree.Count(), wkbs.size());

    auto stats = rtree.Upload({});

    milvus::storage::FileManagerContext ctx_load(
        field_meta_, index_meta_, chunk_manager_);
    ctx_load.set_for_loading_index(true);
    milvus::index::RTreeIndex<std::string> rtree_load(ctx_load);
    nlohmann::json cfg;
    cfg["index_files"] = stats->GetIndexFiles();
    milvus::tracer::TraceContext trace_ctx;
    rtree_load.Load(trace_ctx, cfg);
    ASSERT_EQ(rtree_load.Count(), wkbs.size());
}

TEST_F(RTreeIndexTest, Build_Upload_Load_LargeDataset) {
    // Generate ~10k POINT geometries
    const size_t N = 10000;
    std::vector<std::string> wkbs;
    wkbs.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        // POINT(i i)
        wkbs.emplace_back(CreateWkbFromWkt("POINT(" + std::to_string(i) + " " +
                                           std::to_string(i) + ")"));
    }

    // Write one insert file into remote storage
    auto remote_file = (temp_path_.get() / "geom_large.parquet").string();
    WriteGeometryInsertFile(chunk_manager_, field_meta_, remote_file, wkbs);

    // Build from insert_files (not using BuildWithRawDataForUT)
    milvus::storage::FileManagerContext ctx(
        field_meta_, index_meta_, chunk_manager_);
    milvus::index::RTreeIndex<std::string> rtree(ctx);

    nlohmann::json build_cfg;
    build_cfg["insert_files"] = std::vector<std::string>{remote_file};
    rtree.Build(build_cfg);

    ASSERT_EQ(rtree.Count(), static_cast<int64_t>(N));

    // Upload index
    auto stats = rtree.Upload({});
    ASSERT_GT(stats->GetIndexFiles().size(), 0);

    // Load index back and verify
    milvus::storage::FileManagerContext ctx_load(
        field_meta_, index_meta_, chunk_manager_);
    ctx_load.set_for_loading_index(true);
    milvus::index::RTreeIndex<std::string> rtree_load(ctx_load);

    nlohmann::json cfg_load;
    cfg_load["index_files"] = stats->GetIndexFiles();
    milvus::tracer::TraceContext trace_ctx;
    rtree_load.Load(trace_ctx, cfg_load);

    ASSERT_EQ(rtree_load.Count(), static_cast<int64_t>(N));
}

TEST_F(RTreeIndexTest, Build_BulkLoad_Nulls_And_BadWKB) {
    // five geometries:
    // 1. valid
    // 2. valid but will be marked null
    // 3. valid
    // 4. will be truncated to make invalid
    // 5. valid
    std::vector<std::string> wkbs = {
        CreateWkbFromWkt("POINT(0 0)"),     // valid
        CreateWkbFromWkt("POINT(1 1)"),     // valid 
        CreateWkbFromWkt("POINT(2 2)"),     // valid
        CreatePointWKB(3.0, 3.0),           // will be truncated to make invalid
        CreateWkbFromWkt("POINT(4 4)")      // valid
    };
    // make bad WKB: truncate the 4th geometry
    wkbs[3].resize(wkbs[3].size() / 2);

    // write to remote storage file (chunk manager's root directory)
    auto remote_file = (temp_path_.get() / "geom_bulk.parquet").string();
    WriteGeometryInsertFile(
        chunk_manager_, field_meta_, remote_file, wkbs);

    // build (default to bulk load)
    milvus::storage::FileManagerContext ctx(field_meta_, index_meta_, chunk_manager_);
    milvus::index::RTreeIndex<std::string> rtree(ctx);

    nlohmann::json build_cfg;
    build_cfg["insert_files"] = std::vector<std::string>{remote_file};
    rtree.Build(build_cfg);

    // expect: 3 geometries (0, 2, 4) are valid and parsable, 1st geometry is marked null and skipped, 3rd geometry is bad WKB and skipped
    ASSERT_EQ(rtree.Count(), 4);

    // upload -> load back and verify consistency
    auto stats = rtree.Upload({});
    ASSERT_GT(stats->GetIndexFiles().size(), 0);

    milvus::storage::FileManagerContext ctx_load(field_meta_, index_meta_, chunk_manager_);
    ctx_load.set_for_loading_index(true);
    milvus::index::RTreeIndex<std::string> rtree_load(ctx_load);

    nlohmann::json cfg;
    cfg["index_files"] = stats->GetIndexFiles();

    milvus::tracer::TraceContext trace_ctx;
    rtree_load.Load(trace_ctx, cfg);
    ASSERT_EQ(rtree_load.Count(), 4);
}