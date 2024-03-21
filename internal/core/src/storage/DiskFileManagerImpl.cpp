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

#include <sys/fcntl.h>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/Common.h"
#include "common/Consts.h"
#include "common/EasyAssert.h"
#include "common/FieldData.h"
#include "common/FieldDataInterface.h"
#include "common/File.h"
#include "common/Slice.h"
#include "common/Types.h"
#include "log/Log.h"

#include "storage/DiskFileManagerImpl.h"
#include "storage/FileManager.h"
#include "storage/IndexData.h"
#include "storage/LocalChunkManagerSingleton.h"
#include "storage/ThreadPools.h"
#include "storage/Util.h"

namespace milvus::storage {

DiskFileManagerImpl::DiskFileManagerImpl(
    const FileManagerContext& fileManagerContext,
    std::shared_ptr<milvus_storage::Space> space)
    : FileManagerImpl(fileManagerContext.fieldDataMeta,
                      fileManagerContext.indexMeta),
      space_(space) {
    rcm_ = fileManagerContext.chunkManagerPtr;
}

DiskFileManagerImpl::DiskFileManagerImpl(
    const FileManagerContext& fileManagerContext)
    : FileManagerImpl(fileManagerContext.fieldDataMeta,
                      fileManagerContext.indexMeta) {
    rcm_ = fileManagerContext.chunkManagerPtr;
}

DiskFileManagerImpl::~DiskFileManagerImpl() {
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    local_chunk_manager->RemoveDir(GetIndexPathPrefixWithBuildID(
        local_chunk_manager, index_meta_.build_id));
}

bool
DiskFileManagerImpl::LoadFile(const std::string& file) noexcept {
    return true;
}

std::string
DiskFileManagerImpl::GetRemoteIndexPath(const std::string& file_name,
                                        int64_t slice_num) const {
    std::string remote_prefix;
    if (space_ != nullptr) {
        remote_prefix = GetRemoteIndexObjectPrefixV2();
    } else {
        remote_prefix = GetRemoteIndexObjectPrefix();
    }
    return remote_prefix + "/" + file_name + "_" + std::to_string(slice_num);
}

bool
DiskFileManagerImpl::AddFileUsingSpace(
    const std::string& local_file_name,
    const std::vector<int64_t>& local_file_offsets,
    const std::vector<std::string>& remote_files,
    const std::vector<int64_t>& remote_file_sizes) {
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    for (int64_t i = 0; i < remote_files.size(); ++i) {
        auto buf =
            std::shared_ptr<uint8_t[]>(new uint8_t[remote_file_sizes[i]]);
        local_chunk_manager->Read(local_file_name,
                                  local_file_offsets[i],
                                  buf.get(),
                                  remote_file_sizes[i]);

        auto status =
            space_->WriteBlob(remote_files[i], buf.get(), remote_file_sizes[i]);
        if (!status.ok()) {
            return false;
        }
    }
    return true;
}

bool
DiskFileManagerImpl::AddFile(const std::string& file) noexcept {
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    FILEMANAGER_TRY
    if (!local_chunk_manager->Exist(file)) {
        LOG_ERROR("local file {} not exists", file);
        return false;
    }

    // record local file path
    local_paths_.emplace_back(file);

    auto fileName = GetFileName(file);
    auto fileSize = local_chunk_manager->Size(file);

    std::vector<std::string> batch_remote_files;
    std::vector<int64_t> remote_file_sizes;
    std::vector<int64_t> local_file_offsets;

    int slice_num = 0;
    auto parallel_degree =
        uint64_t(DEFAULT_FIELD_MAX_MEMORY_LIMIT / FILE_SLICE_SIZE);
    for (int64_t offset = 0; offset < fileSize; slice_num++) {
        if (batch_remote_files.size() >= parallel_degree) {
            AddBatchIndexFiles(file,
                               local_file_offsets,
                               batch_remote_files,

                               remote_file_sizes);
            batch_remote_files.clear();
            remote_file_sizes.clear();
            local_file_offsets.clear();
        }

        auto batch_size = std::min(FILE_SLICE_SIZE, int64_t(fileSize) - offset);
        batch_remote_files.emplace_back(
            GetRemoteIndexPath(fileName, slice_num));
        remote_file_sizes.emplace_back(batch_size);
        local_file_offsets.emplace_back(offset);
        offset += batch_size;
    }
    if (batch_remote_files.size() > 0) {
        AddBatchIndexFiles(
            file, local_file_offsets, batch_remote_files, remote_file_sizes);
    }
    FILEMANAGER_CATCH
    FILEMANAGER_END

    return true;
}  // namespace knowhere

void
DiskFileManagerImpl::AddCompactionResultFiles(
    const std::vector<std::string>& files,
    std::unordered_map<std::string, int64_t>& map) {
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    std::vector<std::string> local_files;
    std::vector<std::string> batch_remote_files;
    std::vector<int64_t> remote_file_sizes;
    for (auto i = 0; i < files.size(); ++i) {
        auto file = files[i];
        if (!local_chunk_manager->Exist(file)) {
            LOG_ERROR("local file {} not exists", file);
            std::stringstream err_msg;
            err_msg << "Error: open local file '" << file << " failed, "
                    << strerror(errno);
            throw SegcoreError(FileOpenFailed, err_msg.str());
        }
        auto fileName = GetFileName(file);
        auto fileSize = local_chunk_manager->Size(file);

        auto parallel_degree = 16;

        if (batch_remote_files.size() >= parallel_degree) {
            AddBatchCompactionResultFiles(
                local_files, batch_remote_files, remote_file_sizes, map);
            batch_remote_files.clear();
            remote_file_sizes.clear();
            local_files.clear();
        }
        if (i == 0) {  // centroids file
            batch_remote_files.emplace_back(GetRemoteCentroidsObjectPrefix() +
                                            "/centroids");
        } else {
            batch_remote_files.emplace_back(
                GetRemoteCentroidIdMappingObjectPrefix(fileName) +
                "/offsets_mapping");
        }
        remote_file_sizes.emplace_back(fileSize);
        local_files.emplace_back(file);
    }
    if (batch_remote_files.size() > 0) {
        AddBatchCompactionResultFiles(
            local_files, batch_remote_files, remote_file_sizes, map);
    }
}

void
DiskFileManagerImpl::AddBatchCompactionResultFiles(
    const std::vector<std::string>& local_files,
    const std::vector<std::string>& remote_files,
    const std::vector<int64_t>& remote_file_sizes,
    std::unordered_map<std::string, int64_t>& map) {
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    auto& pool = ThreadPools::GetThreadPool(milvus::ThreadPoolPriority::HIGH);

    std::vector<std::future<std::shared_ptr<uint8_t[]>>> futures;
    futures.reserve(remote_file_sizes.size());

    for (int64_t i = 0; i < remote_files.size(); ++i) {
        futures.push_back(pool.Submit(
            [&](const std::string& file,
                const int64_t data_size) -> std::shared_ptr<uint8_t[]> {
                auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[data_size]);
                local_chunk_manager->Read(file, 0, buf.get(), data_size);
                return buf;
            },
            local_files[i],
            remote_file_sizes[i]));
    }

    std::vector<std::shared_ptr<uint8_t[]>> index_datas;
    std::vector<const uint8_t*> data_slices;
    for (auto& future : futures) {
        auto res = future.get();
        index_datas.emplace_back(res);
        data_slices.emplace_back(res.get());
    }
    PutCompactionResultData(
        rcm_.get(), data_slices, remote_file_sizes, remote_files, map);
}

void
DiskFileManagerImpl::AddBatchIndexFiles(
    const std::string& local_file_name,
    const std::vector<int64_t>& local_file_offsets,
    const std::vector<std::string>& remote_files,
    const std::vector<int64_t>& remote_file_sizes) {
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    auto& pool = ThreadPools::GetThreadPool(milvus::ThreadPoolPriority::HIGH);

    std::vector<std::future<std::shared_ptr<uint8_t[]>>> futures;
    futures.reserve(remote_file_sizes.size());
    AssertInfo(local_file_offsets.size() == remote_files.size(),
               "inconsistent size of offset slices with file slices");
    AssertInfo(remote_files.size() == remote_file_sizes.size(),
               "inconsistent size of file slices with size slices");

    for (int64_t i = 0; i < remote_files.size(); ++i) {
        futures.push_back(pool.Submit(
            [&](const std::string& file,
                const int64_t offset,
                const int64_t data_size) -> std::shared_ptr<uint8_t[]> {
                auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[data_size]);
                local_chunk_manager->Read(file, offset, buf.get(), data_size);
                return buf;
            },
            local_file_name,
            local_file_offsets[i],
            remote_file_sizes[i]));
    }

    // hold index data util upload index file done
    std::vector<std::shared_ptr<uint8_t[]>> index_datas;
    std::vector<const uint8_t*> data_slices;
    for (auto& future : futures) {
        auto res = future.get();
        index_datas.emplace_back(res);
        data_slices.emplace_back(res.get());
    }

    std::map<std::string, int64_t> res;
    if (space_ != nullptr) {
        res = PutIndexData(space_,
                           data_slices,
                           remote_file_sizes,
                           remote_files,
                           field_meta_,
                           index_meta_);
    } else {
        res = PutIndexData(rcm_.get(),
                           data_slices,
                           remote_file_sizes,
                           remote_files,
                           field_meta_,
                           index_meta_);
    }
    for (auto& re : res) {
        remote_paths_to_size_[re.first] = re.second;
    }
}

void
DiskFileManagerImpl::CacheIndexToDisk() {
    auto blobs = space_->StatisticsBlobs();
    std::vector<std::string> remote_files;
    for (auto& blob : blobs) {
        remote_files.push_back(blob.name);
    }
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();

    std::map<std::string, std::vector<int>> index_slices;
    for (auto& file_path : remote_files) {
        auto pos = file_path.find_last_of("_");
        index_slices[file_path.substr(0, pos)].emplace_back(
            std::stoi(file_path.substr(pos + 1)));
    }

    for (auto& slices : index_slices) {
        std::sort(slices.second.begin(), slices.second.end());
    }

    auto EstimateParallelDegree = [&](const std::string& file) -> uint64_t {
        auto fileSize = space_->GetBlobByteSize(file);
        return uint64_t(DEFAULT_FIELD_MAX_MEMORY_LIMIT / fileSize.value());
    };

    for (auto& slices : index_slices) {
        auto prefix = slices.first;
        auto local_index_file_name =
            GetLocalIndexObjectPrefix() +
            prefix.substr(prefix.find_last_of('/') + 1);
        local_chunk_manager->CreateFile(local_index_file_name);
        int64_t offset = 0;
        std::vector<std::string> batch_remote_files;
        uint64_t max_parallel_degree = INT_MAX;
        for (int& iter : slices.second) {
            if (batch_remote_files.size() == max_parallel_degree) {
                auto next_offset = CacheBatchIndexFilesToDiskV2(
                    batch_remote_files, local_index_file_name, offset);
                offset = next_offset;
                batch_remote_files.clear();
            }
            auto origin_file = prefix + "_" + std::to_string(iter);
            if (batch_remote_files.size() == 0) {
                // Use first file size as average size to estimate
                max_parallel_degree = EstimateParallelDegree(origin_file);
            }
            batch_remote_files.push_back(origin_file);
        }
        if (batch_remote_files.size() > 0) {
            auto next_offset = CacheBatchIndexFilesToDiskV2(
                batch_remote_files, local_index_file_name, offset);
            offset = next_offset;
            batch_remote_files.clear();
        }
        local_paths_.emplace_back(local_index_file_name);
    }
}

void
DiskFileManagerImpl::CacheIndexToDisk(
    const std::vector<std::string>& remote_files) {
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();

    std::map<std::string, std::vector<int>> index_slices;
    for (auto& file_path : remote_files) {
        auto pos = file_path.find_last_of('_');
        index_slices[file_path.substr(0, pos)].emplace_back(
            std::stoi(file_path.substr(pos + 1)));
    }

    for (auto& slices : index_slices) {
        std::sort(slices.second.begin(), slices.second.end());
    }

    for (auto& slices : index_slices) {
        auto prefix = slices.first;
        auto local_index_file_name =
            GetLocalIndexObjectPrefix() +
            prefix.substr(prefix.find_last_of('/') + 1);
        local_chunk_manager->CreateFile(local_index_file_name);
        auto file =
            File::Open(local_index_file_name, O_CREAT | O_RDWR | O_TRUNC);

        // Get the remote files
        std::vector<std::string> batch_remote_files;
        batch_remote_files.reserve(slices.second.size());
        for (int& iter : slices.second) {
            auto origin_file = prefix + "_" + std::to_string(iter);
            batch_remote_files.push_back(origin_file);
        }

        auto index_chunks = GetObjectData(rcm_.get(), batch_remote_files);
        for (auto& chunk : index_chunks) {
            auto index_data = chunk.get()->GetFieldData();
            auto index_size = index_data->Size();
            auto chunk_data = reinterpret_cast<uint8_t*>(
                const_cast<void*>(index_data->Data()));
            file.Write(chunk_data, index_size);
        }
        local_paths_.emplace_back(local_index_file_name);
    }
}

uint64_t
DiskFileManagerImpl::CacheBatchIndexFilesToDisk(
    const std::vector<std::string>& remote_files,
    const std::string& local_file_name,
    uint64_t local_file_init_offfset) {
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    auto index_datas = GetObjectData(rcm_.get(), remote_files);
    int batch_size = remote_files.size();
    AssertInfo(index_datas.size() == batch_size,
               "inconsistent file num and index data num!");

    uint64_t offset = local_file_init_offfset;
    for (int i = 0; i < batch_size; ++i) {
        auto index_data = index_datas[i].get()->GetFieldData();
        auto index_size = index_data->Size();
        auto uint8_data =
            reinterpret_cast<uint8_t*>(const_cast<void*>(index_data->Data()));
        local_chunk_manager->Write(
            local_file_name, offset, uint8_data, index_size);
        offset += index_size;
    }
    return offset;
}

uint64_t
DiskFileManagerImpl::CacheBatchIndexFilesToDiskV2(
    const std::vector<std::string>& remote_files,
    const std::string& local_file_name,
    uint64_t local_file_init_offfset) {
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    auto index_datas = GetObjectData(space_, remote_files);
    int batch_size = remote_files.size();
    AssertInfo(index_datas.size() == batch_size,
               "inconsistent file num and index data num!");

    uint64_t offset = local_file_init_offfset;
    for (int i = 0; i < batch_size; ++i) {
        auto index_data = index_datas[i];
        auto index_size = index_data->Size();
        auto uint8_data =
            reinterpret_cast<uint8_t*>(const_cast<void*>(index_data->Data()));
        local_chunk_manager->Write(
            local_file_name, offset, uint8_data, index_size);
        offset += index_size;
    }
    return offset;
}
std::string
DiskFileManagerImpl::CacheRawDataToDisk(
    std::shared_ptr<milvus_storage::Space> space) {
    auto segment_id = GetFieldDataMeta().segment_id;
    auto field_id = GetFieldDataMeta().field_id;

    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    auto local_data_path = storage::GenFieldRawDataPathPrefix(
                               local_chunk_manager, segment_id, field_id) +
                           "raw_data";
    local_chunk_manager->CreateFile(local_data_path);
    // file format
    // num_rows(uint32) | dim(uint32) | index_data ([]uint8_t)
    uint32_t num_rows = 0;
    uint32_t dim = 0;
    int64_t write_offset = sizeof(num_rows) + sizeof(dim);
    auto reader = space->ScanData();
    for (auto rec : *reader) {
        if (!rec.ok()) {
            PanicInfo(IndexBuildError,
                      fmt::format("failed to read data: {}",
                                  rec.status().ToString()));
        }
        auto data = rec.ValueUnsafe();
        if (data == nullptr) {
            break;
        }
        auto total_num_rows = data->num_rows();
        num_rows += total_num_rows;
        auto col_data = data->GetColumnByName(index_meta_.field_name);
        auto field_data = storage::CreateFieldData(
            index_meta_.field_type, index_meta_.dim, total_num_rows);
        field_data->FillFieldData(col_data);
        dim = field_data->get_dim();
        auto data_size =
            field_data->get_num_rows() * index_meta_.dim * sizeof(float);
        local_chunk_manager->Write(local_data_path,
                                   write_offset,
                                   const_cast<void*>(field_data->Data()),
                                   data_size);
        write_offset += data_size;
    }

    // write num_rows and dim value to file header
    write_offset = 0;
    local_chunk_manager->Write(
        local_data_path, write_offset, &num_rows, sizeof(num_rows));
    write_offset += sizeof(num_rows);
    local_chunk_manager->Write(
        local_data_path, write_offset, &dim, sizeof(dim));

    return local_data_path;
}

void
SortByPath(std::vector<std::string>& paths) {
    std::sort(paths.begin(),
              paths.end(),
              [](const std::string& a, const std::string& b) {
                  return std::stol(a.substr(a.find_last_of("/") + 1)) <
                         std::stol(b.substr(b.find_last_of("/") + 1));
              });
}

uint64_t
FetchRawDataAndWriteFile(ChunkManagerPtr rcm,
                         std::string& local_data_path,
                         std::vector<std::string>& batch_files,
                         int64_t& write_offset,
                         uint32_t& num_rows,
                         uint32_t& dim) {
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    auto field_datas = GetObjectData(rcm.get(), batch_files);
    int batch_size = batch_files.size();
    uint64_t batch_data_size = 0;
    for (int i = 0; i < batch_size; ++i) {
        auto field_data = field_datas[i].get()->GetFieldData();
        num_rows += uint32_t(field_data->get_num_rows());
        AssertInfo(dim == 0 || dim == field_data->get_dim(),
                   "inconsistent dim value in multi binlogs!");
        dim = field_data->get_dim();

        auto data_size = field_data->get_num_rows() * dim * sizeof(float);
        local_chunk_manager->Write(local_data_path,
                                   write_offset,
                                   const_cast<void*>(field_data->Data()),
                                   data_size);
        write_offset += data_size;
        batch_data_size += data_size;
    }
    return batch_data_size;
}

// cache raw data for major compaction
uint64_t
DiskFileManagerImpl::CacheCompactionRawDataToDisk(
    const std::map<int64_t, std::vector<std::string>>& remote_files,
    std::vector<std::string>& output_files,
    std::vector<uint64_t>& offsets,
    uint32_t& dim) {
    auto partition_id = GetFieldDataMeta().partition_id;
    auto field_id = GetFieldDataMeta().field_id;

    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    auto local_data_path_prefix =
        storage::GenCompactionRawDataPathPrefix(
            local_chunk_manager, partition_id, field_id) +
        "raw_data";
    auto next_file_id = 0;

    std::vector<std::string> batch_files;

    int64_t write_offset = 0;
    uint32_t num_rows = 0;
    uint64_t whole_size = 0;

    auto parallel_degree =
        uint64_t(DEFAULT_FIELD_MAX_MEMORY_LIMIT / FILE_SLICE_SIZE);

    for (auto& [segment_id, filess] : remote_files) {
        std::vector<std::string> files = filess;
        SortByPath(files);

        auto local_data_path =
            local_data_path_prefix + std::to_string(next_file_id);
        next_file_id++;
        local_chunk_manager->CreateFile(local_data_path);
        output_files.emplace_back(local_data_path);
        batch_files.clear();

        write_offset = 0;
        for (auto& file : files) {
            if (batch_files.size() >= parallel_degree) {
                whole_size += FetchRawDataAndWriteFile(rcm_,
                                                       local_data_path,
                                                       batch_files,
                                                       write_offset,
                                                       num_rows,
                                                       dim);
                batch_files.clear();
            }
            batch_files.emplace_back(file);
        }
        if (batch_files.size() > 0) {
            whole_size += FetchRawDataAndWriteFile(rcm_,
                                                   local_data_path,
                                                   batch_files,
                                                   write_offset,
                                                   num_rows,
                                                   dim);
        }
        offsets.emplace_back(write_offset);
    }

    return whole_size;
}

std::string
DiskFileManagerImpl::CacheRawDataToDisk(std::vector<std::string> remote_files) {
    SortByPath(remote_files);

    auto segment_id = GetFieldDataMeta().segment_id;
    auto field_id = GetFieldDataMeta().field_id;

    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    auto local_data_path = storage::GenFieldRawDataPathPrefix(
                               local_chunk_manager, segment_id, field_id) +
                           "raw_data";
    local_chunk_manager->CreateFile(local_data_path);

    // get batch raw data from s3 and write batch data to disk file
    // TODO: load and write of different batches at the same time
    std::vector<std::string> batch_files;

    // file format
    // num_rows(uint32) | dim(uint32) | index_data ([]uint8_t)
    uint32_t num_rows = 0;
    uint32_t dim = 0;
    int64_t write_offset = sizeof(num_rows) + sizeof(dim);

    auto parallel_degree =
        uint64_t(DEFAULT_FIELD_MAX_MEMORY_LIMIT / FILE_SLICE_SIZE);
    for (auto& file : remote_files) {
        if (batch_files.size() >= parallel_degree) {
            FetchRawDataAndWriteFile(rcm_,
                                     local_data_path,
                                     batch_files,
                                     write_offset,
                                     num_rows,
                                     dim);
            batch_files.clear();
        }

        batch_files.emplace_back(file);
    }

    if (batch_files.size() > 0) {
        FetchRawDataAndWriteFile(
            rcm_, local_data_path, batch_files, write_offset, num_rows, dim);
    }

    // write num_rows and dim value to file header
    write_offset = 0;
    local_chunk_manager->Write(
        local_data_path, write_offset, &num_rows, sizeof(num_rows));
    write_offset += sizeof(num_rows);
    local_chunk_manager->Write(
        local_data_path, write_offset, &dim, sizeof(dim));

    return local_data_path;
}

template <typename T, typename = void>
struct has_native_type : std::false_type {};
template <typename T>
struct has_native_type<T, std::void_t<typename T::NativeType>>
    : std::true_type {};
template <DataType T>
using DataTypeNativeOrVoid =
    typename std::conditional<has_native_type<TypeTraits<T>>::value,
                              typename TypeTraits<T>::NativeType,
                              void>::type;
template <DataType T>
using DataTypeToOffsetMap =
    std::unordered_map<DataTypeNativeOrVoid<T>, int64_t>;

template <DataType T>
void
WriteOptFieldIvfDataImpl(
    const int64_t field_id,
    const std::shared_ptr<LocalChunkManager>& local_chunk_manager,
    const std::string& local_data_path,
    const std::vector<FieldDataPtr>& field_datas,
    uint64_t& write_offset) {
    using FieldDataT = DataTypeNativeOrVoid<T>;
    using OffsetT = uint32_t;
    std::unordered_map<FieldDataT, std::vector<OffsetT>> mp;
    OffsetT offset = 0;
    for (const auto& field_data : field_datas) {
        for (int64_t i = 0; i < field_data->get_num_rows(); ++i) {
            auto val =
                *reinterpret_cast<const FieldDataT*>(field_data->RawValue(i));
            mp[val].push_back(offset++);
        }
    }
    local_chunk_manager->Write(local_data_path,
                               write_offset,
                               const_cast<int64_t*>(&field_id),
                               sizeof(field_id));
    write_offset += sizeof(field_id);
    const uint32_t num_of_unique_field_data = mp.size();
    local_chunk_manager->Write(local_data_path,
                               write_offset,
                               const_cast<uint32_t*>(&num_of_unique_field_data),
                               sizeof(num_of_unique_field_data));
    write_offset += sizeof(num_of_unique_field_data);
    for (const auto& [val, offsets] : mp) {
        const uint32_t offsets_cnt = offsets.size();
        local_chunk_manager->Write(local_data_path,
                                   write_offset,
                                   const_cast<uint32_t*>(&offsets_cnt),
                                   sizeof(offsets_cnt));
        write_offset += sizeof(offsets_cnt);
        const size_t data_size = offsets_cnt * sizeof(OffsetT);
        local_chunk_manager->Write(local_data_path,
                                   write_offset,
                                   const_cast<OffsetT*>(offsets.data()),
                                   data_size);
        write_offset += data_size;
    }
}

#define GENERATE_OPT_FIELD_IVF_IMPL(DT)               \
    WriteOptFieldIvfDataImpl<DT>(field_id,            \
                                 local_chunk_manager, \
                                 local_data_path,     \
                                 field_datas,         \
                                 write_offset)
bool
WriteOptFieldIvfData(
    const DataType& dt,
    const int64_t field_id,
    const std::shared_ptr<LocalChunkManager>& local_chunk_manager,
    const std::string& local_data_path,
    const std::vector<FieldDataPtr>& field_datas,
    uint64_t& write_offset) {
    switch (dt) {
        case DataType::BOOL:
            GENERATE_OPT_FIELD_IVF_IMPL(DataType::BOOL);
            break;
        case DataType::INT8:
            GENERATE_OPT_FIELD_IVF_IMPL(DataType::INT8);
            break;
        case DataType::INT16:
            GENERATE_OPT_FIELD_IVF_IMPL(DataType::INT16);
            break;
        case DataType::INT32:
            GENERATE_OPT_FIELD_IVF_IMPL(DataType::INT32);
            break;
        case DataType::INT64:
            GENERATE_OPT_FIELD_IVF_IMPL(DataType::INT64);
            break;
        case DataType::FLOAT:
            GENERATE_OPT_FIELD_IVF_IMPL(DataType::FLOAT);
            break;
        case DataType::DOUBLE:
            GENERATE_OPT_FIELD_IVF_IMPL(DataType::DOUBLE);
            break;
        case DataType::STRING:
            GENERATE_OPT_FIELD_IVF_IMPL(DataType::STRING);
            break;
        case DataType::VARCHAR:
            GENERATE_OPT_FIELD_IVF_IMPL(DataType::VARCHAR);
            break;
        default:
            LOG_WARN("Unsupported data type in optional scalar field: ", dt);
            return false;
    }
    return true;
}
#undef GENERATE_OPT_FIELD_IVF_IMPL

void
WriteOptFieldsIvfMeta(
    const std::shared_ptr<LocalChunkManager>& local_chunk_manager,
    const std::string& local_data_path,
    const uint32_t num_of_fields,
    uint64_t& write_offset) {
    const uint8_t kVersion = 0;
    local_chunk_manager->Write(local_data_path,
                               write_offset,
                               const_cast<uint8_t*>(&kVersion),
                               sizeof(kVersion));
    write_offset += sizeof(kVersion);
    local_chunk_manager->Write(local_data_path,
                               write_offset,
                               const_cast<uint32_t*>(&num_of_fields),
                               sizeof(num_of_fields));
    write_offset += sizeof(num_of_fields);
}

// write optional scalar fields ivf info in the following format without space among them
// | (meta)
// | version (uint8_t) | num_of_fields (uint32_t) |
// | (field_0)
// | field_id (int64_t) | num_of_unique_field_data (uint32_t)
// | size_0 (uint32_t) | offset_0 (uint32_t)...
// | size_1 | offset_0, offset_1, ...
std::string
DiskFileManagerImpl::CacheOptFieldToDisk(
    std::shared_ptr<milvus_storage::Space> space, OptFieldT& fields_map) {
    uint32_t num_of_fields = fields_map.size();
    if (0 == num_of_fields) {
        return "";
    } else if (num_of_fields > 1) {
        PanicInfo(
            ErrorCode::NotImplemented,
            "vector index build with multiple fields is not supported yet");
    }
    if (nullptr == space) {
        LOG_ERROR("Failed to cache optional field. Space is null");
        return "";
    }

    auto segment_id = GetFieldDataMeta().segment_id;
    auto vec_field_id = GetFieldDataMeta().field_id;
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    auto local_data_path = storage::GenFieldRawDataPathPrefix(
                               local_chunk_manager, segment_id, vec_field_id) +
                           std::string(VEC_OPT_FIELDS);
    local_chunk_manager->CreateFile(local_data_path);

    uint64_t write_offset = 0;
    WriteOptFieldsIvfMeta(
        local_chunk_manager, local_data_path, num_of_fields, write_offset);

    auto reader = space->ScanData();
    for (auto& [field_id, tup] : fields_map) {
        const auto& field_name = std::get<0>(tup);
        const auto& field_type = std::get<1>(tup);
        std::vector<FieldDataPtr> field_datas;
        for (auto rec : *reader) {
            if (!rec.ok()) {
                PanicInfo(IndexBuildError,
                          fmt::format("failed to read optional field data: {}",
                                      rec.status().ToString()));
            }
            auto data = rec.ValueUnsafe();
            if (data == nullptr) {
                break;
            }
            auto total_num_rows = data->num_rows();
            if (0 == total_num_rows) {
                LOG_WARN("optional field {} has no data", field_name);
                return "";
            }
            auto col_data = data->GetColumnByName(field_name);
            auto field_data =
                storage::CreateFieldData(field_type, 1, total_num_rows);
            field_data->FillFieldData(col_data);
            field_datas.emplace_back(field_data);
        }
        if (!WriteOptFieldIvfData(field_type,
                                  field_id,
                                  local_chunk_manager,
                                  local_data_path,
                                  field_datas,
                                  write_offset)) {
            return "";
        }
    }
    return local_data_path;
}

std::string
DiskFileManagerImpl::CacheOptFieldToDisk(OptFieldT& fields_map) {
    uint32_t num_of_fields = fields_map.size();
    if (0 == num_of_fields) {
        return "";
    } else if (num_of_fields > 1) {
        PanicInfo(
            ErrorCode::NotImplemented,
            "vector index build with multiple fields is not supported yet");
    }

    auto segment_id = GetFieldDataMeta().segment_id;
    auto vec_field_id = GetFieldDataMeta().field_id;
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    auto local_data_path = storage::GenFieldRawDataPathPrefix(
                               local_chunk_manager, segment_id, vec_field_id) +
                           std::string(VEC_OPT_FIELDS);
    local_chunk_manager->CreateFile(local_data_path);

    std::vector<FieldDataPtr> field_datas;
    std::vector<std::string> batch_files;
    uint64_t write_offset = 0;
    WriteOptFieldsIvfMeta(
        local_chunk_manager, local_data_path, num_of_fields, write_offset);

    auto FetchRawData = [&]() {
        auto fds = GetObjectData(rcm_.get(), batch_files);
        for (size_t i = 0; i < batch_files.size(); ++i) {
            auto data = fds[i].get()->GetFieldData();
            field_datas.emplace_back(data);
        }
    };

    auto parallel_degree =
        uint64_t(DEFAULT_FIELD_MAX_MEMORY_LIMIT / FILE_SLICE_SIZE);
    for (auto& [field_id, tup] : fields_map) {
        const auto& field_type = std::get<1>(tup);
        auto& field_paths = std::get<2>(tup);
        if (0 == field_paths.size()) {
            LOG_WARN("optional field {} has no data", field_id);
            return "";
        }

        std::vector<FieldDataPtr>().swap(field_datas);
        SortByPath(field_paths);

        for (auto& file : field_paths) {
            if (batch_files.size() >= parallel_degree) {
                FetchRawData();
                batch_files.clear();
            }
            batch_files.emplace_back(file);
        }
        if (batch_files.size() > 0) {
            FetchRawData();
        }
        if (!WriteOptFieldIvfData(field_type,
                                  field_id,
                                  local_chunk_manager,
                                  local_data_path,
                                  field_datas,
                                  write_offset)) {
            return "";
        }
    }
    return local_data_path;
}

std::string
DiskFileManagerImpl::GetFileName(const std::string& localfile) {
    boost::filesystem::path localPath(localfile);
    return localPath.filename().string();
}

std::string
DiskFileManagerImpl::GetLocalIndexObjectPrefix() {
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    return GenIndexPathPrefix(
        local_chunk_manager, index_meta_.build_id, index_meta_.index_version);
}

std::string
DiskFileManagerImpl::GetLocalRawDataObjectPrefix() {
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    return GenFieldRawDataPathPrefix(
        local_chunk_manager, field_meta_.segment_id, field_meta_.field_id);
}

// need to confirm the raw data path, used for train
std::string
DiskFileManagerImpl::GetCompactionRawDataObjectPrefix() {
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    return GenCompactionRawDataPathPrefix(
        local_chunk_manager, field_meta_.partition_id, field_meta_.field_id);
}

// need to confirm the result path, used for data partition and search
std::string
DiskFileManagerImpl::GetCompactionResultObjectPrefix() {
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    return GenCompactionResultPathPrefix(
        local_chunk_manager, index_meta_.build_id, index_meta_.index_version);
}

bool
DiskFileManagerImpl::RemoveFile(const std::string& file) noexcept {
    // TODO: implement this interface
    return false;
}

std::optional<bool>
DiskFileManagerImpl::IsExisted(const std::string& file) noexcept {
    bool isExist = false;
    auto local_chunk_manager =
        LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    try {
        isExist = local_chunk_manager->Exist(file);
    } catch (std::exception& e) {
        // LOG_DEBUG("Exception:{}", e).what();
        return std::nullopt;
    }
    return isExist;
}

}  // namespace milvus::storage
