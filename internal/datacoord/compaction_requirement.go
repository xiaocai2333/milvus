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
	"context"

	"github.com/milvus-io/milvus/internal/util/taskresource"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

// compactionInputFromMeta sizes a compaction's inputs from DataCoord's own
// metadata, which is the side that has the right answer.
//
// It reads getSegmentSize() and getDeltaLogSize() rather than the per-field
// binlog arrays, and that distinction is the whole point on storage v3: those
// arrays are not persisted for v3 segments at all -- kv_catalog.go's
// buildAlterSegmentsKvs skips them ("V3 segments persist paths via the LOON
// manifest") -- so a DataNode reconstructing the requirement from the plan's
// arrays after a DataCoord restart computes zero and falls to the estimator's
// 64MiB floor. SegmentInfo.Stats, which getSegmentSize goes through, is
// populated either way.
//
// allResolved is false when any input segment could not be fetched. The
// estimate is still usable -- it just under-counts the missing ones -- but
// callers must not cache it, or a transient meta miss becomes this task's
// permanent requirement.
func compactionInputFromMeta(ctx context.Context, meta CompactionMeta, taskID int64,
	compactionType datapb.CompactionType, inputSegments []int64,
) (taskresource.CompactionInput, bool) {
	var totalMemory, totalRows, maxDelete, sumDelete, storageVersion int64
	allResolved := true

	for _, segID := range inputSegments {
		segment := meta.GetHealthySegment(ctx, segID)
		if segment == nil {
			allResolved = false
			mlog.Warn(ctx, "could not resolve input segment for resource estimation, estimate will under-count it",
				mlog.Int64("taskID", taskID),
				mlog.String("compactionType", compactionType.String()),
				mlog.Int64("segmentID", segID))
			continue
		}
		totalMemory += segment.getSegmentSize()
		totalRows += segment.GetNumOfRows()

		d := segment.getDeltaLogSize()
		sumDelete += d
		if d > maxDelete {
			maxDelete = d
		}

		// Storage version is per segment, not per task: neither CompactionTask
		// nor CompactionPlan carries one. Take the max, matching what
		// RequirementForCompaction does on the DataNode side -- a v3 segment in
		// a mixed plan dominates the memory profile, so the max is the
		// conservative direction.
		if v := segment.GetStorageVersion(); v > storageVersion {
			storageVersion = v
		}
	}

	// L0's ComposeDeleteDataFromSegments loads every segment's deletes at once,
	// while the streaming compactors' ComposeDeleteFromDeltalogs holds one
	// segment's at a time. The estimator takes one field for both, so the caller's
	// type decides which aggregation is the right one to hand it.
	deleteBytes := maxDelete
	if compactionType == datapb.CompactionType_Level0DeleteCompaction {
		deleteBytes = sumDelete
	}

	return taskresource.CompactionInput{
		Type:                  compactionType,
		StorageVersion:        storageVersion,
		TotalMemorySize:       totalMemory,
		TotalRows:             totalRows,
		MaxSegmentDeleteBytes: deleteBytes,
	}, allResolved
}

// compactionRequirementFromMeta is compactionInputFromMeta run through the
// estimator. allResolved is returned so a caller that caches knows whether it
// may.
func compactionRequirementFromMeta(ctx context.Context, meta CompactionMeta, taskID int64,
	compactionType datapb.CompactionType, inputSegments []int64,
) (taskresource.Requirement, bool) {
	in, allResolved := compactionInputFromMeta(ctx, meta, taskID, compactionType, inputSegments)
	return taskresource.EstimateCompaction(in), allResolved
}
