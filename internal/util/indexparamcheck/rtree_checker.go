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

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

// RTREEChecker checks if a RTREE index can be built.
type RTREEChecker struct {
	scalarIndexChecker
}

func (c *RTREEChecker) CheckTrain(dataType schemapb.DataType, params map[string]string) error {
	if !typeutil.IsGeometryType(dataType) {
		return fmt.Errorf("RTREE index can only be built on geometry field")
	}

	// Set default values if not provided
	setDefaultIfNotExist(params, RTreeFillFactor, fmt.Sprintf("%f", DefaultRTreeFillFactor))
	setDefaultIfNotExist(params, RTreeIndexCapacity, fmt.Sprintf("%d", DefaultRTreeIndexCapacity))
	setDefaultIfNotExist(params, RTreeLeafCapacity, fmt.Sprintf("%d", DefaultRTreeLeafCapacity))
	setDefaultIfNotExist(params, RTreeDim, fmt.Sprintf("%d", DefaultRTreeDim))
	setDefaultIfNotExist(params, RTreeRV, DefaultRTreeRV)

	// Validate fillFactor
	if !CheckFloatByRange(params, RTreeFillFactor, MinRTreeFillFactor, MaxRTreeFillFactor) {
		return errOutOfRange(params[RTreeFillFactor], MinRTreeFillFactor, MaxRTreeFillFactor)
	}

	// Validate indexCapacity
	if !CheckIntByRange(params, RTreeIndexCapacity, MinRTreeIndexCapacity, MaxRTreeIndexCapacity) {
		return errOutOfRange(params[RTreeIndexCapacity], MinRTreeIndexCapacity, MaxRTreeIndexCapacity)
	}

	// Validate leafCapacity
	if !CheckIntByRange(params, RTreeLeafCapacity, MinRTreeLeafCapacity, MaxRTreeLeafCapacity) {
		return errOutOfRange(params[RTreeLeafCapacity], MinRTreeLeafCapacity, MaxRTreeLeafCapacity)
	}

	// Validate dim
	if !CheckIntByRange(params, RTreeDim, MinRTreeDim, MaxRTreeDim) {
		return errOutOfRange(params[RTreeDim], MinRTreeDim, MaxRTreeDim)
	}

	// Validate rv
	rvValue, exists := params[RTreeRV]
	if exists {
		validRVValues := []string{"RV_LINEAR", "RV_QUADRATIC", "RV_RSTAR"}
		if !funcutil.SliceContain(validRVValues, rvValue) {
			return fmt.Errorf("rv value %s is not supported, supported values: %v", rvValue, validRVValues)
		}
	}

	return c.scalarIndexChecker.CheckTrain(dataType, params)
}

func (c *RTREEChecker) CheckValidDataType(indexType IndexType, field *schemapb.FieldSchema) error {
	dType := field.GetDataType()
	if !typeutil.IsGeometryType(dType) {
		return fmt.Errorf("RTREE index can only be built on geometry field, got %s", dType.String())
	}
	return nil
}

func newRTREEChecker() *RTREEChecker {
	return &RTREEChecker{}
}
