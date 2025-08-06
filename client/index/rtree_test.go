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
	"testing"

	"github.com/stretchr/testify/suite"
)

type RTreeIndexSuite struct {
	suite.Suite
}

func (s *RTreeIndexSuite) TestNewRTreeIndex() {
	idx := NewRTreeIndex()
	s.Equal(RTREE, idx.IndexType())

	params := idx.Params()
	s.Equal(string(RTREE), params[IndexTypeKey])
	s.Equal(strconv.FormatFloat(DefaultRTreeFillFactor, 'f', -1, 64), params[RTreeFillFactorKey])
	s.Equal(strconv.Itoa(DefaultRTreeIndexCapacity), params[RTreeIndexCapacityKey])
	s.Equal(strconv.Itoa(DefaultRTreeLeafCapacity), params[RTreeLeafCapacityKey])
	s.Equal(strconv.Itoa(DefaultRTreeDim), params[RTreeDimKey])
	s.Equal(DefaultRTreeRV, params[RTreeRVKey])
}

func (s *RTreeIndexSuite) TestNewRTreeIndexWithParams() {
	fillFactor := 0.7
	indexCapacity := 150
	leafCapacity := 150
	dim := 3
	rv := "RV_LINEAR"

	idx := NewRTreeIndexWithParams(fillFactor, indexCapacity, leafCapacity, dim, rv)
	s.Equal(RTREE, idx.IndexType())

	params := idx.Params()
	s.Equal(string(RTREE), params[IndexTypeKey])
	s.Equal(strconv.FormatFloat(fillFactor, 'f', -1, 64), params[RTreeFillFactorKey])
	s.Equal(strconv.Itoa(indexCapacity), params[RTreeIndexCapacityKey])
	s.Equal(strconv.Itoa(leafCapacity), params[RTreeLeafCapacityKey])
	s.Equal(strconv.Itoa(dim), params[RTreeDimKey])
	s.Equal(rv, params[RTreeRVKey])
}

func (s *RTreeIndexSuite) TestRTreeIndexBuilder() {
	idx := NewRTreeIndexBuilder().
		WithFillFactor(0.6).
		WithIndexCapacity(200).
		WithLeafCapacity(200).
		WithDimension(2).
		WithRVType("RV_QUADRATIC").
		Build()

	s.Equal(RTREE, idx.IndexType())

	params := idx.Params()
	s.Equal(string(RTREE), params[IndexTypeKey])
	s.Equal("0.6", params[RTreeFillFactorKey])
	s.Equal("200", params[RTreeIndexCapacityKey])
	s.Equal("200", params[RTreeLeafCapacityKey])
	s.Equal("2", params[RTreeDimKey])
	s.Equal("RV_QUADRATIC", params[RTreeRVKey])
}

func (s *RTreeIndexSuite) TestRTreeIndexBuilderDefaults() {
	idx := NewRTreeIndexBuilder().Build()
	s.Equal(RTREE, idx.IndexType())

	params := idx.Params()
	s.Equal(string(RTREE), params[IndexTypeKey])
	s.Equal(strconv.FormatFloat(DefaultRTreeFillFactor, 'f', -1, 64), params[RTreeFillFactorKey])
	s.Equal(strconv.Itoa(DefaultRTreeIndexCapacity), params[RTreeIndexCapacityKey])
	s.Equal(strconv.Itoa(DefaultRTreeLeafCapacity), params[RTreeLeafCapacityKey])
	s.Equal(strconv.Itoa(DefaultRTreeDim), params[RTreeDimKey])
	s.Equal(DefaultRTreeRV, params[RTreeRVKey])
}

func (s *RTreeIndexSuite) TestRTreeIndexBuilderChaining() {
	builder := NewRTreeIndexBuilder()

	// Test method chaining
	result := builder.
		WithFillFactor(0.9).
		WithIndexCapacity(50).
		WithLeafCapacity(25).
		WithDimension(3).
		WithRVType("RV_RSTAR")

	s.Equal(builder, result) // Should return the same builder instance

	idx := result.Build()
	params := idx.Params()
	s.Equal("0.9", params[RTreeFillFactorKey])
	s.Equal("50", params[RTreeIndexCapacityKey])
	s.Equal("25", params[RTreeLeafCapacityKey])
	s.Equal("3", params[RTreeDimKey])
	s.Equal("RV_RSTAR", params[RTreeRVKey])
}

func TestRTreeIndex(t *testing.T) {
	suite.Run(t, new(RTreeIndexSuite))
}
