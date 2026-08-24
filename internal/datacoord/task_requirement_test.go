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

package datacoord

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/util/taskresource"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/taskcommon"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

const reqGiB = int64(1) << 30

// The flat-charge families: neither reads segment data, so the requirement must
// not vary with the collection, but it must also never be zero -- zero means
// "unknown" to the scheduler and would take the task out of the accounting.
func TestFlatChargeTasksReportANonZeroRequirement(t *testing.T) {
	paramtable.Init()

	refresh := &refreshExternalCollectionTask{times: taskcommon.NewTimes()}
	got := refresh.GetResourceRequirement()
	assert.Positive(t, got.Memory, "a flat charge is still a charge; zero would read as unknown")

	copyTask := &copySegmentTask{times: taskcommon.NewTimes()}
	copyTask.task.Store(&datapb.CopySegmentTask{
		IdMappings: []*datapb.CopySegmentIDMapping{{}, {}, {}},
	})
	gotCopy := copyTask.GetResourceRequirement()
	assert.Positive(t, gotCopy.Memory)
	assert.Positive(t, gotCopy.CPU, "concurrency scales with how many copies run at once")
}

// A copy of more segments costs more CPU (more concurrent copies) but the same
// memory, because CopyObject is server-side and no segment bytes cross the
// worker.
func TestCopySegmentScalesCPUNotMemory(t *testing.T) {
	paramtable.Init()

	mk := func(n int) taskresource.Requirement {
		task := &copySegmentTask{times: taskcommon.NewTimes()}
		mappings := make([]*datapb.CopySegmentIDMapping, n)
		for i := range mappings {
			mappings[i] = &datapb.CopySegmentIDMapping{}
		}
		task.task.Store(&datapb.CopySegmentTask{IdMappings: mappings})
		return task.GetResourceRequirement()
	}

	few, many := mk(1), mk(64)
	assert.Equal(t, few.Memory, many.Memory, "no segment bytes pass through the worker")
	assert.Greater(t, many.CPU, few.CPU)
}

// An index build is sized from the field it targets. A bigger field must cost
// more; an unresolvable segment must report UNKNOWN rather than free.
func TestIndexBuildUnresolvableSegmentIsUnknown(t *testing.T) {
	paramtable.Init()

	m := &meta{segments: NewSegmentsInfo(), indexMeta: newSegmentIndexMeta(nil)}
	task := &indexBuildTask{
		SegmentIndex: &model.SegmentIndex{SegmentID: 1, CollectionID: 100, IndexID: 10, BuildID: 7},
		meta:         m,
		times:        taskcommon.NewTimes(),
	}

	// With no segment in meta the task must say "unknown", not "free".
	assert.Equal(t, taskresource.Requirement{}, task.GetResourceRequirement(),
		"an unresolvable segment is unknown, and the scheduler falls back to node state")
}

// A Sort sub-job arriving as a stats request is a sort compaction wearing a
// different hat, and must be priced as one -- EstimateStats says so explicitly
// and refuses to handle it.
func TestStatsSortSubJobIsPricedAsCompaction(t *testing.T) {
	paramtable.Init()

	m := &meta{segments: NewSegmentsInfo()}
	m.segments.SetSegment(1, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 1, State: commonpb.SegmentState_Flushed, NumOfRows: 1_000_000,
		StorageVersion: 3,
	}})

	sortTask := &statsTask{
		StatsTask: &indexpb.StatsTask{SegmentID: 1, SubJobType: indexpb.StatsSubJob_Sort},
		meta:      m, times: taskcommon.NewTimes(),
	}
	text := &statsTask{
		StatsTask: &indexpb.StatsTask{SegmentID: 1, SubJobType: indexpb.StatsSubJob_TextIndexJob},
		meta:      m, times: taskcommon.NewTimes(),
	}

	// Both resolve; what matters is that they take different estimators, so the
	// figures differ for the same segment.
	gotSort := sortTask.GetResourceRequirement()
	gotText := text.GetResourceRequirement()
	assert.Positive(t, gotSort.Memory)
	assert.Positive(t, gotText.Memory)
	assert.NotEqual(t, gotSort.Memory, gotText.Memory,
		"a Sort sub-job is a compaction and must not be priced with the text-index factor")
}

// An unresolvable segment must yield the zero Requirement everywhere, so the
// scheduler treats the task as unmeasured rather than as needing nothing.
func TestUnresolvableSegmentReportsUnknown(t *testing.T) {
	paramtable.Init()

	m := &meta{segments: NewSegmentsInfo()}
	st := &statsTask{
		StatsTask: &indexpb.StatsTask{SegmentID: 404, SubJobType: indexpb.StatsSubJob_TextIndexJob},
		meta:      m, times: taskcommon.NewTimes(),
	}
	assert.Equal(t, taskresource.Requirement{}, st.GetResourceRequirement())
}

// Analyze is the largest memory consumer of any DataNode task: it allocates a
// fraction of the WHOLE NODE regardless of how much data exists. The charge has
// to reflect that, or the guard stops serializing it the way the old
// 65535-slot constant did.
func TestAnalyzeChargesTheNodeFraction(t *testing.T) {
	paramtable.Init()

	m := &meta{segments: NewSegmentsInfo()}
	m.segments.SetSegment(1, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 1, State: commonpb.SegmentState_Flushed, NumOfRows: 10_000_000,
	}})

	at := &analyzeTask{
		AnalyzeTask: &indexpb.AnalyzeTask{
			SegmentIDs: []int64{1}, Dim: 768, FieldType: schemapb.DataType_FloatVector,
		},
		meta: m, times: taskcommon.NewTimes(),
	}
	got := at.GetResourceRequirement()
	require.Positive(t, got.Memory)
	assert.Greater(t, got.Memory, reqGiB,
		"a 10M-row 768-dim training set is multi-gigabyte; a small figure means the ratio was lost")
}
