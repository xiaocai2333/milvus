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

#include <gtest/gtest.h>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/type.h>

#include "common/Types.h"
#include "storage/Util.h"

using namespace milvus;
using namespace milvus::storage;

class NormalizeVectorArraysTest : public ::testing::Test {};

TEST_F(NormalizeVectorArraysTest, ListFloatToFixedSizeBinary) {
    // Simulate external parquet data: List<Float32> with dim=4
    const int dim = 4;
    const int num_rows = 3;
    float raw_data[num_rows][dim] = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f},
        {9.0f, 10.0f, 11.0f, 12.0f},
    };

    // Build a ListArray<Float32>
    auto value_builder = std::make_shared<arrow::FloatBuilder>();
    arrow::ListBuilder list_builder(arrow::default_memory_pool(), value_builder);

    for (int i = 0; i < num_rows; i++) {
        ASSERT_TRUE(list_builder.Append().ok());
        for (int j = 0; j < dim; j++) {
            ASSERT_TRUE(value_builder->Append(raw_data[i][j]).ok());
        }
    }

    std::shared_ptr<arrow::Array> list_array;
    ASSERT_TRUE(list_builder.Finish(&list_array).ok());
    ASSERT_EQ(list_array->type_id(), arrow::Type::LIST);
    ASSERT_EQ(list_array->length(), num_rows);

    // Normalize to FixedSizeBinary
    auto result = NormalizeVectorArraysToFixedSizeBinary(
        {list_array}, DataType::VECTOR_FLOAT, dim);

    ASSERT_EQ(result.size(), 1);
    auto& normalized = result[0];
    ASSERT_EQ(normalized->type_id(), arrow::Type::FIXED_SIZE_BINARY);
    ASSERT_EQ(normalized->length(), num_rows);

    auto fsb_array =
        std::static_pointer_cast<arrow::FixedSizeBinaryArray>(normalized);
    ASSERT_EQ(fsb_array->byte_width(), dim * sizeof(float));

    // Verify data integrity
    for (int i = 0; i < num_rows; i++) {
        auto value = fsb_array->GetView(i);
        const float* floats = reinterpret_cast<const float*>(value.data());
        for (int j = 0; j < dim; j++) {
            ASSERT_FLOAT_EQ(floats[j], raw_data[i][j]);
        }
    }
}

TEST_F(NormalizeVectorArraysTest, FixedSizeListFloatToFixedSizeBinary) {
    // Simulate FixedSizeList<Float32> with dim=4
    const int dim = 4;
    const int num_rows = 2;
    float raw_data[num_rows][dim] = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f},
    };

    auto value_builder = std::make_shared<arrow::FloatBuilder>();
    arrow::FixedSizeListBuilder fsl_builder(
        arrow::default_memory_pool(), value_builder, dim);

    for (int i = 0; i < num_rows; i++) {
        ASSERT_TRUE(fsl_builder.Append().ok());
        for (int j = 0; j < dim; j++) {
            ASSERT_TRUE(value_builder->Append(raw_data[i][j]).ok());
        }
    }

    std::shared_ptr<arrow::Array> fsl_array;
    ASSERT_TRUE(fsl_builder.Finish(&fsl_array).ok());
    ASSERT_EQ(fsl_array->type_id(), arrow::Type::FIXED_SIZE_LIST);

    auto result = NormalizeVectorArraysToFixedSizeBinary(
        {fsl_array}, DataType::VECTOR_FLOAT, dim);

    ASSERT_EQ(result.size(), 1);
    auto& normalized = result[0];
    ASSERT_EQ(normalized->type_id(), arrow::Type::FIXED_SIZE_BINARY);
    ASSERT_EQ(normalized->length(), num_rows);

    auto fsb_array =
        std::static_pointer_cast<arrow::FixedSizeBinaryArray>(normalized);
    for (int i = 0; i < num_rows; i++) {
        auto value = fsb_array->GetView(i);
        const float* floats = reinterpret_cast<const float*>(value.data());
        for (int j = 0; j < dim; j++) {
            ASSERT_FLOAT_EQ(floats[j], raw_data[i][j]);
        }
    }
}

TEST_F(NormalizeVectorArraysTest, AlreadyFixedSizeBinaryPassthrough) {
    // FixedSizeBinary should pass through unchanged
    const int dim = 4;
    const int byte_width = dim * sizeof(float);
    const int num_rows = 2;
    float raw_data[num_rows][dim] = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f},
    };

    arrow::FixedSizeBinaryBuilder builder(arrow::fixed_size_binary(byte_width));
    for (int i = 0; i < num_rows; i++) {
        ASSERT_TRUE(
            builder
                .Append(reinterpret_cast<const uint8_t*>(raw_data[i]),
                        byte_width)
                .ok());
    }

    std::shared_ptr<arrow::Array> fsb_array;
    ASSERT_TRUE(builder.Finish(&fsb_array).ok());

    auto result = NormalizeVectorArraysToFixedSizeBinary(
        {fsb_array}, DataType::VECTOR_FLOAT, dim);

    ASSERT_EQ(result.size(), 1);
    // Should be the same pointer (passthrough)
    ASSERT_EQ(result[0].get(), fsb_array.get());
}
