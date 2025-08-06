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

package indexparamcheck

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

func TestRTREEChecker(t *testing.T) {
	c := newRTREEChecker()

	t.Run("valid data type", func(t *testing.T) {
		field := &schemapb.FieldSchema{
			DataType: schemapb.DataType_Geometry,
		}
		err := c.CheckValidDataType(IndexRTREE, field)
		assert.NoError(t, err)
	})

	t.Run("invalid data type", func(t *testing.T) {
		field := &schemapb.FieldSchema{
			DataType: schemapb.DataType_VarChar,
		}
		err := c.CheckValidDataType(IndexRTREE, field)
		assert.Error(t, err)
	})

	t.Run("valid parameters with defaults", func(t *testing.T) {
		params := make(map[string]string)
		err := c.CheckTrain(schemapb.DataType_Geometry, params)
		assert.NoError(t, err)

		// Check that defaults are applied
		assert.Equal(t, fmt.Sprintf("%f", DefaultRTreeFillFactor), params[RTreeFillFactor])
		assert.Equal(t, strconv.Itoa(DefaultRTreeIndexCapacity), params[RTreeIndexCapacity])
		assert.Equal(t, strconv.Itoa(DefaultRTreeLeafCapacity), params[RTreeLeafCapacity])
		assert.Equal(t, strconv.Itoa(DefaultRTreeDim), params[RTreeDim])
		assert.Equal(t, DefaultRTreeRV, params[RTreeRV])
	})

	t.Run("valid custom parameters", func(t *testing.T) {
		params := map[string]string{
			RTreeFillFactor:    "0.7",
			RTreeIndexCapacity: "150",
			RTreeLeafCapacity:  "150",
			RTreeDim:           "2",
			RTreeRV:            "RV_LINEAR",
		}
		err := c.CheckTrain(schemapb.DataType_Geometry, params)
		assert.NoError(t, err)
	})

	t.Run("invalid fillFactor - too low", func(t *testing.T) {
		params := map[string]string{
			RTreeFillFactor: "0.05",
		}
		err := c.CheckTrain(schemapb.DataType_Geometry, params)
		assert.Error(t, err)
	})

	t.Run("invalid fillFactor - too high", func(t *testing.T) {
		params := map[string]string{
			RTreeFillFactor: "1.5",
		}
		err := c.CheckTrain(schemapb.DataType_Geometry, params)
		assert.Error(t, err)
	})

	t.Run("invalid indexCapacity - too low", func(t *testing.T) {
		params := map[string]string{
			RTreeIndexCapacity: "1",
		}
		err := c.CheckTrain(schemapb.DataType_Geometry, params)
		assert.Error(t, err)
	})

	t.Run("invalid indexCapacity - too high", func(t *testing.T) {
		params := map[string]string{
			RTreeIndexCapacity: "1500",
		}
		err := c.CheckTrain(schemapb.DataType_Geometry, params)
		assert.Error(t, err)
	})

	t.Run("invalid leafCapacity - too low", func(t *testing.T) {
		params := map[string]string{
			RTreeLeafCapacity: "1",
		}
		err := c.CheckTrain(schemapb.DataType_Geometry, params)
		assert.Error(t, err)
	})

	t.Run("invalid leafCapacity - too high", func(t *testing.T) {
		params := map[string]string{
			RTreeLeafCapacity: "1500",
		}
		err := c.CheckTrain(schemapb.DataType_Geometry, params)
		assert.Error(t, err)
	})

	t.Run("invalid dimension - too low", func(t *testing.T) {
		params := map[string]string{
			RTreeDim: "1",
		}
		err := c.CheckTrain(schemapb.DataType_Geometry, params)
		assert.Error(t, err)
	})

	t.Run("invalid dimension - too high", func(t *testing.T) {
		params := map[string]string{
			RTreeDim: "4",
		}
		err := c.CheckTrain(schemapb.DataType_Geometry, params)
		assert.Error(t, err)
	})

	t.Run("invalid rv value", func(t *testing.T) {
		params := map[string]string{
			RTreeRV: "INVALID_RV",
		}
		err := c.CheckTrain(schemapb.DataType_Geometry, params)
		assert.Error(t, err)
	})

	t.Run("valid rv values", func(t *testing.T) {
		validRVs := []string{"RV_LINEAR", "RV_QUADRATIC", "RV_RSTAR"}
		for _, rv := range validRVs {
			params := map[string]string{
				RTreeRV: rv,
			}
			err := c.CheckTrain(schemapb.DataType_Geometry, params)
			assert.NoError(t, err)
		}
	})

	t.Run("non-geometry data type", func(t *testing.T) {
		params := make(map[string]string)
		err := c.CheckTrain(schemapb.DataType_VarChar, params)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "RTREE index can only be built on geometry field")
	})
}
