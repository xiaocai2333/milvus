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

package index

import (
	"strconv"
)

// RTree index parameter keys
const (
	RTreeFillFactorKey    = "fillFactor"
	RTreeIndexCapacityKey = "indexCapacity"
	RTreeLeafCapacityKey  = "leafCapacity"
	RTreeDimKey           = "dim"
	RTreeRVKey            = "rv"
)

// RTree index parameter defaults
const (
	DefaultRTreeFillFactor    = 0.8
	DefaultRTreeIndexCapacity = 100
	DefaultRTreeLeafCapacity  = 100
	DefaultRTreeDim           = 2
	DefaultRTreeRV            = "RV_RSTAR"
)

var _ Index = rtreeIndex{}

// rtreeIndex represents an RTree index for geometry fields
type rtreeIndex struct {
	baseIndex
	fillFactor    float64
	indexCapacity int
	leafCapacity  int
	dim           int
	rv            string
}

func (idx rtreeIndex) Params() map[string]string {
	params := map[string]string{
		IndexTypeKey:          string(RTREE),
		RTreeFillFactorKey:    strconv.FormatFloat(idx.fillFactor, 'f', -1, 64),
		RTreeIndexCapacityKey: strconv.Itoa(idx.indexCapacity),
		RTreeLeafCapacityKey:  strconv.Itoa(idx.leafCapacity),
		RTreeDimKey:           strconv.Itoa(idx.dim),
		RTreeRVKey:            idx.rv,
	}
	return params
}

// NewRTreeIndex creates a new RTree index with default parameters
func NewRTreeIndex() Index {
	return rtreeIndex{
		baseIndex: baseIndex{
			indexType: RTREE,
		},
		fillFactor:    DefaultRTreeFillFactor,
		indexCapacity: DefaultRTreeIndexCapacity,
		leafCapacity:  DefaultRTreeLeafCapacity,
		dim:           DefaultRTreeDim,
		rv:            DefaultRTreeRV,
	}
}

// NewRTreeIndexWithParams creates a new RTree index with custom parameters
func NewRTreeIndexWithParams(fillFactor float64, indexCapacity, leafCapacity, dim int, rv string) Index {
	return rtreeIndex{
		baseIndex: baseIndex{
			indexType: RTREE,
		},
		fillFactor:    fillFactor,
		indexCapacity: indexCapacity,
		leafCapacity:  leafCapacity,
		dim:           dim,
		rv:            rv,
	}
}

// RTreeIndexBuilder provides a fluent API for building RTree indexes
type RTreeIndexBuilder struct {
	index rtreeIndex
}

// NewRTreeIndexBuilder creates a new RTree index builder
func NewRTreeIndexBuilder() *RTreeIndexBuilder {
	return &RTreeIndexBuilder{
		index: rtreeIndex{
			baseIndex: baseIndex{
				indexType: RTREE,
			},
			fillFactor:    DefaultRTreeFillFactor,
			indexCapacity: DefaultRTreeIndexCapacity,
			leafCapacity:  DefaultRTreeLeafCapacity,
			dim:           DefaultRTreeDim,
			rv:            DefaultRTreeRV,
		},
	}
}

// WithFillFactor sets the fill factor for the RTree index
func (b *RTreeIndexBuilder) WithFillFactor(fillFactor float64) *RTreeIndexBuilder {
	b.index.fillFactor = fillFactor
	return b
}

// WithIndexCapacity sets the index capacity for the RTree index
func (b *RTreeIndexBuilder) WithIndexCapacity(capacity int) *RTreeIndexBuilder {
	b.index.indexCapacity = capacity
	return b
}

// WithLeafCapacity sets the leaf capacity for the RTree index
func (b *RTreeIndexBuilder) WithLeafCapacity(capacity int) *RTreeIndexBuilder {
	b.index.leafCapacity = capacity
	return b
}

// WithDimension sets the dimension for the RTree index
func (b *RTreeIndexBuilder) WithDimension(dim int) *RTreeIndexBuilder {
	b.index.dim = dim
	return b
}

// WithRVType sets the RV type for the RTree index
// Valid values: "RV_LINEAR", "RV_QUADRATIC", "RV_RSTAR"
func (b *RTreeIndexBuilder) WithRVType(rv string) *RTreeIndexBuilder {
	b.index.rv = rv
	return b
}

// Build returns the constructed RTree index
func (b *RTreeIndexBuilder) Build() Index {
	return b.index
}
