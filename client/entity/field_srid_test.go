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

package entity

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFieldWithSrid(t *testing.T) {
	t.Run("WithSrid should set SRID parameter", func(t *testing.T) {
		field := NewField().WithName("geometry_field").WithDataType(FieldTypeGeometry).WithSrid(3857)

		srid, err := field.GetSrid()
		assert.NoError(t, err)
		assert.Equal(t, int32(3857), srid)

		// Check that the TypeParam was set correctly
		assert.Equal(t, "3857", field.TypeParams[TypeParamSrid])
	})

	t.Run("GetSrid should return default 4326 when not set", func(t *testing.T) {
		field := NewField().WithName("geometry_field").WithDataType(FieldTypeGeometry)

		srid, err := field.GetSrid()
		assert.NoError(t, err)
		assert.Equal(t, int32(4326), srid)
	})

	t.Run("GetSrid should handle invalid SRID gracefully", func(t *testing.T) {
		field := NewField().WithName("geometry_field").WithDataType(FieldTypeGeometry)
		field.TypeParams[TypeParamSrid] = "invalid"

		srid, err := field.GetSrid()
		assert.Error(t, err)
		assert.Equal(t, int32(4326), srid) // Should return default on error
		assert.Contains(t, err.Error(), "bad format srid")
	})

	t.Run("WithSrid should work with zero SRID", func(t *testing.T) {
		field := NewField().WithName("geometry_field").WithDataType(FieldTypeGeometry).WithSrid(0)

		srid, err := field.GetSrid()
		assert.NoError(t, err)
		assert.Equal(t, int32(0), srid)
	})

	t.Run("WithSrid should work with large SRID values", func(t *testing.T) {
		field := NewField().WithName("geometry_field").WithDataType(FieldTypeGeometry).WithSrid(32767)

		srid, err := field.GetSrid()
		assert.NoError(t, err)
		assert.Equal(t, int32(32767), srid)
	})

	t.Run("GetSrid should reject negative SRID", func(t *testing.T) {
		field := NewField().WithName("geometry_field").WithDataType(FieldTypeGeometry)
		field.TypeParams[TypeParamSrid] = "-1"

		srid, err := field.GetSrid()
		assert.Error(t, err)
		assert.Equal(t, int32(4326), srid) // Should return default on error
		assert.Contains(t, err.Error(), "SRID must be non-negative")
	})

	t.Run("GetSrid should reject too large SRID", func(t *testing.T) {
		field := NewField().WithName("geometry_field").WithDataType(FieldTypeGeometry)
		field.TypeParams[TypeParamSrid] = "2147483648" // 2^31, exceeds 32-bit signed int

		srid, err := field.GetSrid()
		assert.Error(t, err)
		assert.Equal(t, int32(4326), srid) // Should return default on error
		assert.Contains(t, err.Error(), "SRID value too large")
	})

	t.Run("GetSrid should accept maximum valid SRID", func(t *testing.T) {
		field := NewField().WithName("geometry_field").WithDataType(FieldTypeGeometry)
		field.TypeParams[TypeParamSrid] = "2147483647" // 2^31 - 1, maximum 32-bit signed int

		srid, err := field.GetSrid()
		assert.NoError(t, err)
		assert.Equal(t, int32(2147483647), srid)
	})

	t.Run("ProtoMessage should include SRID in TypeParams", func(t *testing.T) {
		field := NewField().WithName("geometry_field").WithDataType(FieldTypeGeometry).WithSrid(4326)

		proto := field.ProtoMessage()

		// Check that SRID is in TypeParams
		sridFound := false
		for _, param := range proto.TypeParams {
			if param.Key == TypeParamSrid {
				assert.Equal(t, "4326", param.Value)
				sridFound = true
				break
			}
		}
		assert.True(t, sridFound, "SRID should be in TypeParams")
	})
}
