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

package proxy

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/common"
)

func TestValidateGeometryField(t *testing.T) {
	t.Run("geometry field without SRID should get default 4326", func(t *testing.T) {
		field := &schemapb.FieldSchema{
			Name:       "geometry_field",
			DataType:   schemapb.DataType_Geometry,
			TypeParams: []*commonpb.KeyValuePair{},
		}

		err := validateGeometryField(field)
		assert.NoError(t, err)

		// Check that default SRID was added
		sridFound := false
		for _, param := range field.TypeParams {
			if param.Key == common.SridKey {
				assert.Equal(t, "4326", param.Value)
				sridFound = true
				break
			}
		}
		assert.True(t, sridFound, "Default SRID should be added")
	})

	t.Run("geometry field with valid SRID should pass validation", func(t *testing.T) {
		field := &schemapb.FieldSchema{
			Name:     "geometry_field",
			DataType: schemapb.DataType_Geometry,
			TypeParams: []*commonpb.KeyValuePair{
				{Key: common.SridKey, Value: "3857"},
			},
		}

		err := validateGeometryField(field)
		assert.NoError(t, err)

		// Check that custom SRID was preserved
		sridFound := false
		for _, param := range field.TypeParams {
			if param.Key == common.SridKey {
				assert.Equal(t, "3857", param.Value)
				sridFound = true
				break
			}
		}
		assert.True(t, sridFound, "Custom SRID should be preserved")
	})

	t.Run("geometry field with invalid SRID should fail validation", func(t *testing.T) {
		field := &schemapb.FieldSchema{
			Name:     "geometry_field",
			DataType: schemapb.DataType_Geometry,
			TypeParams: []*commonpb.KeyValuePair{
				{Key: common.SridKey, Value: "invalid_srid"},
			},
		}

		err := validateGeometryField(field)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "invalid SRID value")
	})

	t.Run("geometry field with negative SRID should fail validation", func(t *testing.T) {
		field := &schemapb.FieldSchema{
			Name:     "geometry_field",
			DataType: schemapb.DataType_Geometry,
			TypeParams: []*commonpb.KeyValuePair{
				{Key: common.SridKey, Value: "-1"},
			},
		}

		err := validateGeometryField(field)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "SRID must be non-negative")
	})

	t.Run("geometry field with too large SRID should fail validation", func(t *testing.T) {
		field := &schemapb.FieldSchema{
			Name:     "geometry_field",
			DataType: schemapb.DataType_Geometry,
			TypeParams: []*commonpb.KeyValuePair{
				{Key: common.SridKey, Value: "2147483648"}, // 2^31, exceeds 32-bit signed int
			},
		}

		err := validateGeometryField(field)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "SRID value too large")
	})

	t.Run("geometry field with maximum valid SRID should pass validation", func(t *testing.T) {
		field := &schemapb.FieldSchema{
			Name:     "geometry_field",
			DataType: schemapb.DataType_Geometry,
			TypeParams: []*commonpb.KeyValuePair{
				{Key: common.SridKey, Value: "2147483647"}, // 2^31 - 1, maximum 32-bit signed int
			},
		}

		err := validateGeometryField(field)
		assert.NoError(t, err)
	})

	t.Run("geometry field with non-numeric SRID should fail validation", func(t *testing.T) {
		field := &schemapb.FieldSchema{
			Name:     "geometry_field",
			DataType: schemapb.DataType_Geometry,
			TypeParams: []*commonpb.KeyValuePair{
				{Key: common.SridKey, Value: "abc123"},
			},
		}

		err := validateGeometryField(field)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "SRID must be a valid integer")
	})

	t.Run("geometry field with floating point SRID should fail validation", func(t *testing.T) {
		field := &schemapb.FieldSchema{
			Name:     "geometry_field",
			DataType: schemapb.DataType_Geometry,
			TypeParams: []*commonpb.KeyValuePair{
				{Key: common.SridKey, Value: "4326.5"},
			},
		}

		err := validateGeometryField(field)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "SRID must be a valid integer")
	})

	t.Run("geometry field with zero SRID should pass validation", func(t *testing.T) {
		field := &schemapb.FieldSchema{
			Name:     "geometry_field",
			DataType: schemapb.DataType_Geometry,
			TypeParams: []*commonpb.KeyValuePair{
				{Key: common.SridKey, Value: "0"},
			},
		}

		err := validateGeometryField(field)
		assert.NoError(t, err)
	})
}

func TestValidateFieldTypeWithGeometry(t *testing.T) {
	t.Run("schema with geometry field should be validated", func(t *testing.T) {
		schema := &schemapb.CollectionSchema{
			Name: "test_collection",
			Fields: []*schemapb.FieldSchema{
				{
					Name:         "pk",
					DataType:     schemapb.DataType_Int64,
					IsPrimaryKey: true,
				},
				{
					Name:     "geometry_field",
					DataType: schemapb.DataType_Geometry,
					TypeParams: []*commonpb.KeyValuePair{
						{Key: common.SridKey, Value: "4326"},
					},
				},
			},
		}

		err := validateFieldType(schema)
		assert.NoError(t, err)
	})

	t.Run("schema with geometry field without SRID should get default", func(t *testing.T) {
		schema := &schemapb.CollectionSchema{
			Name: "test_collection",
			Fields: []*schemapb.FieldSchema{
				{
					Name:         "pk",
					DataType:     schemapb.DataType_Int64,
					IsPrimaryKey: true,
				},
				{
					Name:       "geometry_field",
					DataType:   schemapb.DataType_Geometry,
					TypeParams: []*commonpb.KeyValuePair{},
				},
			},
		}

		err := validateFieldType(schema)
		assert.NoError(t, err)

		// Check that default SRID was added
		geometryField := schema.Fields[1]
		sridFound := false
		for _, param := range geometryField.TypeParams {
			if param.Key == common.SridKey {
				assert.Equal(t, "4326", param.Value)
				sridFound = true
				break
			}
		}
		assert.True(t, sridFound, "Default SRID should be added")
	})
}
