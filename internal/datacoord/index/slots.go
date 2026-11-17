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
	"math"

	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func calculateIndexTaskSlot(fieldSize int64, isVectorIndex bool) int64 {
	if isVectorIndex {
		slotsPerUnit := max(Params.DataNodeCfg.WorkerSlotUnit.GetAsInt64(), 1) *
			max(Params.DataNodeCfg.BuildParallel.GetAsInt64(), 1)
		if paramtable.GetRole() == typeutil.StandaloneRole {
			slotsPerUnit = max(int64(float64(slotsPerUnit)*Params.DataNodeCfg.StandaloneSlotRatio.GetAsFloat()), 1)
		}
		bytesPerSlot := max(int64(8<<30)/slotsPerUnit, 1)
		peakBytes := int64(math.MaxInt64)
		if fieldSize <= math.MaxInt64/4 {
			peakBytes = max(fieldSize, 0) * 4
		}
		return max(peakBytes/bytesPerSlot+min(peakBytes%bytesPerSlot, 1), 1)
	}
	defaultSlots := Params.DataCoordCfg.ScalarIndexTaskSlotUsage.GetAsInt64()
	if fieldSize > 512*1024*1024 {
		taskSlot := max(fieldSize/512/1024/1024, 1) * defaultSlots
		return max(taskSlot, 1)
	} else if fieldSize > 100*1024*1024 {
		return max(defaultSlots/4, 1)
	} else if fieldSize > 10*1024*1024 {
		return max(defaultSlots/16, 1)
	}
	return max(defaultSlots/64, 1)
}
