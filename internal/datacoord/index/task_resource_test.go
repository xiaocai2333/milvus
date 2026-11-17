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
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/datacoord/meta"
	"github.com/milvus-io/milvus/internal/datacoord/session"
	globalTask "github.com/milvus-io/milvus/internal/datacoord/task"
	"github.com/milvus-io/milvus/internal/datacoord/taskresource"
	catalogmocks "github.com/milvus-io/milvus/internal/metastore/mocks"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/taskcommon"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

// fixtureIndexMeta adds an HNSW index on the fixture's vector field to mt.
func fixtureIndexMeta(t *testing.T, mt *meta.Meta) (*meta.Meta, *IndexMeta) {
	catalog := catalogmocks.NewDataCoordCatalog(t)
	catalog.EXPECT().AlterSegmentIndexes(mock.Anything, mock.Anything).Return(nil).Maybe()
	im := createIndexMetaWithSegment(catalog, taskresource.FixtureCollectionID, taskresource.FixturePartitionID,
		taskresource.FixtureSegmentID, 4, 101, 5)
	mt.IndexCatalog = im
	return mt, im
}

func bigFixtureIndexTask(mt *meta.Meta, im *IndexMeta) *indexBuildTask {
	return newIndexBuildTask(&model.SegmentIndex{
		CollectionID: 1, PartitionID: 2, SegmentID: 3, IndexID: 4, BuildID: 5, NumRows: taskresource.BigFixtureRows,
	}, 1, mt, im, nil, nil, nil)
}

func TestTaskResource_Index(t *testing.T) {
	paramtable.Init()
	mt, im := fixtureIndexMeta(t, taskresource.NewFixtureMeta(false))
	segIndex := &model.SegmentIndex{CollectionID: 1, PartitionID: 2, SegmentID: 3, IndexID: 4, BuildID: 5, NumRows: 1000}
	it := newIndexBuildTask(segIndex, 1, mt, im, nil, nil, nil)
	// HNSW on a 128-dim float vector over 1000 rows with no binlogs: closed form.
	assert.Equal(t, taskresource.Index(1000*128*4, true), taskresource.Price(it.GetTaskResource()))
	// Cached: the value the scheduler placed on is the value the request ships.
	assert.Equal(t, taskresource.Index(1000*128*4, true), taskresource.Price(it.GetTaskResource()))

	// Segment gone: floor, not cached.
	orphan := newIndexBuildTask(&model.SegmentIndex{CollectionID: 1, SegmentID: 999, IndexID: 4, BuildID: 6}, 1, mt, im, nil, nil, nil)
	assert.Equal(t, taskresource.Default(), taskresource.Price(orphan.GetTaskResource()))
	assert.Equal(t, taskresource.Default(), taskresource.Price(orphan.GetTaskResource()))
}

// TestTaskResource_IndexScalar covers a non-vector index: CPU falls back to the
// default and the field bytes are what the segment holds beyond its
// fixed-width fields -- exactly 64 bytes per row here, the varchar being the
// only variable-width field.
func TestTaskResource_IndexScalar(t *testing.T) {
	paramtable.Init()
	mt, im := fixtureIndexMeta(t, taskresource.NewFixtureMeta(false))
	// Repoint the index at the varchar field with an inverted (non-vector) index.
	idx := im.Indexes[1][4]
	idx.FieldID = 102
	idx.IndexParams = []*commonpb.KeyValuePair{{Key: "index_type", Value: "INVERTED"}}

	it := newIndexBuildTask(&model.SegmentIndex{CollectionID: 1, PartitionID: 2, SegmentID: 3, IndexID: 4, BuildID: 5, NumRows: 1000}, 1, mt, im, nil, nil, nil)
	assert.Equal(t, taskresource.Index(1000*64, false), taskresource.Price(it.GetTaskResource()))
}

// TestTaskResource_IndexSchemaCacheMiss is the case that must never be cached:
// a V3 segment (no per-field binlogs, Stats present) whose collection schema is
// not in the cache yet. Without a schema there is no way to tell the indexed
// field from the rest of the segment, so a cached answer here would freeze a
// whole-segment over-estimate on a one-field build for the task's lifetime.
//
// The answer is that whole-segment over-estimate, not the floor. The scheduler
// places the task on it, so it must bound the build from above.
func TestTaskResource_IndexSchemaCacheMiss(t *testing.T) {
	paramtable.Init()
	mt, im := fixtureIndexMeta(t, taskresource.NewFixtureMeta(true))
	// The segment keeps its V3 shape (Binlogs empty, Stats non-zero); only the
	// collection is missing, exactly as it is before the cache is warmed.
	taskresource.DropFixtureSchema(mt)
	assert.Greater(t, mt.Segments.GetSegment(3).GetSegmentSize(), int64(0))
	assert.Empty(t, mt.Segments.GetSegment(3).GetBinlogs())

	priced := taskresource.Index(taskresource.BigFixtureRows*128*4, true)
	// The index type is unknown too while the schema is missing, so the bound
	// takes the scalar CPU request. Only memory decides whether a worker fits.
	bound := taskresource.Index(taskresource.BigFixtureSegmentSize, true)
	bound.CPU = taskresource.DefaultCPU()
	assert.NotEqual(t, taskresource.Default(), bound, "the bug this test guards must be observable")

	it := bigFixtureIndexTask(mt, im)
	assert.Equal(t, bound, taskresource.Price(it.GetTaskResource()))
	assert.False(t, taskresource.Resolved(it.GetTaskResource()), "the scheduler must be told the price is not exact")
	assert.Equal(t, bound, taskresource.Price(it.GetTaskResource()))
	assert.Greater(t, bound.Memory, priced.Memory, "the fallback must err high, never low")

	// Not cached: once the schema arrives the very next call reprices from meta,
	// which proves the closure ran again instead of serving a frozen value.
	taskresource.PutFixtureSchema(mt, taskresource.FixtureSchema())
	assert.Equal(t, priced, taskresource.Price(it.GetTaskResource()))
}

// TestTaskResource_IndexEmptyIndexParams covers an index whose params have not
// been read back yet: there is no index type, so "is this a vector index" has no
// answer and a vector build would otherwise be frozen at the scalar CPU request.
// Note GetIndexType answers the invalidIndex sentinel here, not "".
func TestTaskResource_IndexEmptyIndexParams(t *testing.T) {
	paramtable.Init()
	mt, im := fixtureIndexMeta(t, taskresource.NewFixtureMeta(true))
	idx := im.Indexes[1][4]
	idx.IndexParams = nil

	priced := taskresource.Index(taskresource.BigFixtureRows*128*4, true)
	misPriced := taskresource.Index(taskresource.BigFixtureRows*128*4, false)
	assert.NotEqual(t, taskresource.Default(), misPriced, "the bug this test guards must be observable")
	// The schema is known here, but the index type is not, so the price is the
	// whole segment at the scalar CPU request until the params are read back.
	bound := taskresource.Index(taskresource.BigFixtureSegmentSize, true)
	bound.CPU = taskresource.DefaultCPU()

	it := bigFixtureIndexTask(mt, im)
	assert.Equal(t, bound, taskresource.Price(it.GetTaskResource()))
	assert.Equal(t, bound, taskresource.Price(it.GetTaskResource()))
	assert.Greater(t, bound.Memory, priced.Memory, "the fallback must err high, never low")

	// Not cached: the params arriving repairs the price on the next call.
	idx.IndexParams = []*commonpb.KeyValuePair{{Key: common.IndexTypeKey, Value: "HNSW"}}
	assert.Equal(t, priced, taskresource.Price(it.GetTaskResource()))
}

// TestTaskResource_IndexFieldNotInSchema is the other side of the same coin: the
// schema IS known and the field is genuinely not in it. That is a real answer,
// not a transient miss, so the conservative whole-segment price is cached.
func TestTaskResource_IndexFieldNotInSchema(t *testing.T) {
	paramtable.Init()
	mt, im := fixtureIndexMeta(t, taskresource.NewFixtureMeta(false))
	idx := im.Indexes[1][4]
	idx.FieldID = 999

	it := newIndexBuildTask(&model.SegmentIndex{CollectionID: 1, PartitionID: 2, SegmentID: 3, IndexID: 4, BuildID: 5, NumRows: 1000}, 1, mt, im, nil, nil, nil)
	assert.Equal(t, taskresource.Index(taskresource.FixtureSegmentSize, true), taskresource.Price(it.GetTaskResource()))

	// Liveness is checked before the cache: a dropped target must not retain
	// the old whole-segment price.
	mt.Segments.DropSegment(3)
	assert.Equal(t, taskresource.Default(), taskresource.Price(it.GetTaskResource()))
	assert.False(t, taskresource.Resolved(it.GetTaskResource()))
}

// TestTaskResource_IndexUnpriceableField covers a known schema whose field still
// sizes to nothing (an empty segment): the floor, and not cached.
func TestTaskResource_IndexUnpriceableField(t *testing.T) {
	paramtable.Init()
	mt, im := fixtureIndexMeta(t, taskresource.NewFixtureMeta(false))
	taskresource.UpdateFixtureSegment(mt, func(seg *datapb.SegmentInfo) {
		seg.Stats = nil
		seg.NumOfRows = 0
	})

	it := newIndexBuildTask(&model.SegmentIndex{CollectionID: 1, SegmentID: 3, IndexID: 4, BuildID: 5, NumRows: 1000}, 1, mt, im, nil, nil, nil)
	assert.Equal(t, taskresource.Default(), taskresource.Price(it.GetTaskResource()))
	assert.Equal(t, taskresource.Default(), taskresource.Price(it.GetTaskResource()))
}

func TestTaskResource_DroppedIndexBypassesCachedPrice(t *testing.T) {
	paramtable.Init()
	mt, im := fixtureIndexMeta(t, taskresource.NewFixtureMeta(true))
	it := bigFixtureIndexTask(mt, im)
	assert.Greater(t, taskresource.Price(it.GetTaskResource()).Memory, taskresource.Default().Memory)
	im.Indexes[1][4].IsDeleted = true
	assert.Equal(t, taskresource.Default(), taskresource.Price(it.GetTaskResource()))
	assert.False(t, taskresource.Resolved(it.GetTaskResource()))
	// A reconstructed task after a coordinator restart must agree too.
	assert.Equal(t, taskresource.Default(), taskresource.Price(bigFixtureIndexTask(mt, im).GetTaskResource()))
}

// Use real index metadata/pricing and the running global scheduler, rather than
// mocking a dropped task as already priced at the floor.
func TestIndexScheduler_CoordinatorOnlyTasksDoNotConsumeWorkerMemory(t *testing.T) {
	paramtable.Init()
	for _, name := range []string{"FLAT", "BIN_FLAT", "small", "nullable", "dropped-segment", "dropped-index", "dropped-index-reload"} {
		t.Run(name, func(t *testing.T) {
			mt, im := fixtureIndexMeta(t, taskresource.NewFixtureMeta(true))
			segIndex, ok := im.GetIndexJob(5)
			require.True(t, ok)
			segIndex.NumRows = taskresource.BigFixtureRows
			im.SegmentBuildInfo.Add(segIndex)
			handler := meta.NewNMockHandler(t)
			collectionInfo := mt.GetCollection(1)
			handler.EXPECT().GetCollection(mock.Anything, int64(1)).Return(collectionInfo, nil).Maybe()
			it := newIndexBuildTask(segIndex, 128, mt, im, handler, nil, nil)
			const capacity = int64(64 << 20)
			require.Greater(t, taskresource.Price(it.GetTaskResource()).Memory, capacity)
			wantState := commonpb.IndexState_Finished
			switch name {
			case "FLAT", "BIN_FLAT":
				im.Indexes[1][4].IndexParams = []*commonpb.KeyValuePair{{Key: common.IndexTypeKey, Value: name}}
			case "small":
				segIndex.NumRows = 1
				im.SegmentBuildInfo.Add(segIndex)
			case "nullable":
				collectionInfo.Schema.Fields[1].Nullable = true
				taskresource.UpdateFixtureSegment(mt, func(seg *datapb.SegmentInfo) {
					seg.Stats.NullCounts = map[int64]int64{101: taskresource.BigFixtureRows}
				})
			case "dropped-segment":
				mt.Segments.DropSegment(3)
				wantState = commonpb.IndexState_Failed
			case "dropped-index", "dropped-index-reload":
				im.Indexes[1][4].IsDeleted = true
				wantState = commonpb.IndexState_Failed
				if name == "dropped-index-reload" {
					it = newIndexBuildTask(segIndex, 128, mt, im, handler, nil, nil)
				}
			}
			cluster := session.NewMockCluster(t)
			cluster.EXPECT().QuerySlot().Return(map[int64]*session.WorkerSlots{
				1: {NodeID: 1, TotalMemory: capacity, AvailableMemory: capacity},
			}).Maybe()
			scheduler := globalTask.NewGlobalTaskScheduler(context.Background(), cluster)
			var dispatched atomic.Bool
			follower := globalTask.NewMockTask(t)
			follower.EXPECT().GetTaskID().Return(int64(6)).Maybe()
			follower.EXPECT().GetTaskType().Return(taskcommon.Compaction).Maybe()
			follower.EXPECT().GetTaskSlot().Return(int64(128)).Maybe()
			follower.EXPECT().GetTaskResource().Return(taskcommon.Resource{CPU: 1, Memory: capacity}, true).Maybe()
			follower.EXPECT().GetTaskState().RunAndReturn(func() taskcommon.State {
				if dispatched.Load() {
					return taskcommon.Finished
				}
				return taskcommon.Init
			}).Maybe()
			follower.EXPECT().SetTaskTime(mock.Anything, mock.Anything).Return().Maybe()
			follower.EXPECT().CreateTaskOnWorker(int64(1), cluster).Run(func(int64, session.Cluster) { dispatched.Store(true) }).Once()
			scheduler.Enqueue(it)
			scheduler.Enqueue(follower)
			scheduler.Start()
			// Join scheduler callbacks before the test and mock expectations finish.
			defer scheduler.Stop()
			require.Eventually(t, func() bool {
				persisted, ok := im.GetIndexJob(5)
				return ok && persisted.IndexState == wantState && dispatched.Load()
			}, 5*time.Second, 10*time.Millisecond)
			cluster.AssertNotCalled(t, "CreateIndex", mock.Anything, mock.Anything, mock.Anything)
		})
	}
}
