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

package compaction

import (
	"context"
	"fmt"
	sio "io"
	"sort"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"go.opentelemetry.io/otel"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/internal/datanode/allocator"
	"github.com/milvus-io/milvus/internal/datanode/io"
	iter "github.com/milvus-io/milvus/internal/datanode/iterators"
	"github.com/milvus-io/milvus/internal/metastore/kv/binlog"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/timerecord"
	"github.com/milvus-io/milvus/pkg/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

// for SortingCompaction only
type sortingCompactionTask struct {
	binlogIO io.BinlogIO
	allocator.Allocator
	currentTs typeutil.Timestamp

	plan *datapb.CompactionPlan

	ctx    context.Context
	cancel context.CancelFunc

	done chan struct{}
	tr   *timerecord.TimeRecorder
}

// make sure compactionTask implements compactor interface
var _ Compactor = (*sortingCompactionTask)(nil)

func NewSortingCompactionTask(
	ctx context.Context,
	binlogIO io.BinlogIO,
	alloc allocator.Allocator,
	plan *datapb.CompactionPlan,
) *sortingCompactionTask {
	ctx1, cancel := context.WithCancel(ctx)
	return &sortingCompactionTask{
		ctx:       ctx1,
		cancel:    cancel,
		binlogIO:  binlogIO,
		Allocator: alloc,
		plan:      plan,
		tr:        timerecord.NewTimeRecorder("mix compaction"),
		currentTs: tsoutil.GetCurrentTime(),
		done:      make(chan struct{}, 1),
	}
}

func (t *sortingCompactionTask) Complete() {
	t.done <- struct{}{}
}

func (t *sortingCompactionTask) Stop() {
	t.cancel()
	<-t.done
}

func (t *sortingCompactionTask) GetPlanID() typeutil.UniqueID {
	return t.plan.GetPlanID()
}

func (t *sortingCompactionTask) GetCollection() typeutil.UniqueID {
	return t.plan.GetSegmentBinlogs()[0].GetCollectionID()
}

func (t *sortingCompactionTask) GetChannelName() string {
	return t.plan.GetChannel()
}

func (t *sortingCompactionTask) getNumRows() int64 {
	numRows := int64(0)
	if len(t.plan.GetSegmentBinlogs()[0].GetFieldBinlogs()) > 0 {
		for _, b := range t.plan.GetSegmentBinlogs()[0].GetFieldBinlogs()[0].GetBinlogs() {
			numRows += b.GetEntriesNum()
		}
	}
	return numRows
}

func (t *sortingCompactionTask) Compact() (*datapb.CompactionPlanResult, error) {
	durInQueue := t.tr.RecordSpan()
	compactStart := time.Now()
	ctx, span := otel.Tracer(typeutil.DataNodeRole).Start(t.ctx, fmt.Sprintf("MixCompact-%d", t.GetPlanID()))
	defer span.End()

	if len(t.plan.GetSegmentBinlogs()) != 1 {
		log.Warn("sorting compaction wrong, there's not one segments in segment binlogs",
			zap.Int64("planID", t.plan.GetPlanID()), zap.Int("segment num", len(t.plan.GetSegmentBinlogs())))
		return nil, errors.New("compaction plan is illegal")
	}

	collectionID := t.plan.GetSegmentBinlogs()[0].GetCollectionID()
	partitionID := t.plan.GetSegmentBinlogs()[0].GetPartitionID()
	segmentID := t.plan.GetSegmentBinlogs()[0].GetSegmentID()
	log := log.With(zap.Int64("collectionID", collectionID),
		zap.Int64("partitionID", partitionID),
		zap.Int64("segmentID", segmentID))

	if ok := funcutil.CheckCtxValid(ctx); !ok {
		log.Warn("compact wrong, task context done or timeout")
		return nil, ctx.Err()
	}
	ctx, cancelAll := context.WithTimeout(ctx, time.Duration(t.plan.GetTimeoutInSeconds())*time.Second)
	defer cancelAll()

	targetSegID, err := t.AllocOne()
	if err != nil {
		log.Warn("compact wrong, unable to allocate segmentID", zap.Error(err))
		return nil, err
	}

	log.Info("sorting compaction start", zap.Int64("target segmentID", targetSegID))

	numRows := t.getNumRows()
	writer, err := NewSegmentWriter(t.plan.GetSchema(), numRows, targetSegID, partitionID, collectionID)
	if err != nil {
		log.Warn("sort segment wrong, unable to init segment writer", zap.Error(err))
		return nil, err
	}

	var (
		syncBatchCount    int   // binlog batch count
		unflushedRowCount int64 = 0

		// All binlog meta of a segment
		allBinlogs = make(map[typeutil.UniqueID]*datapb.FieldBinlog)
	)

	serWriteTimeCost := time.Duration(0)
	uploadTimeCost := time.Duration(0)
	sortTimeCost := time.Duration(0)

	values, err := t.downloadData(ctx, numRows, writer.GetPkID())
	if err != nil {
		log.Warn("download data failed", zap.Error(err))
		return nil, nil
	}

	sortStart := time.Now()
	sort.Slice(values, func(i, j int) bool {
		return values[i].PK.LT(values[j].PK)
	})
	sortTimeCost += time.Since(sortStart)

	for _, v := range values {
		err := writer.Write(v)
		if err != nil {
			log.Warn("compact wrong, failed to writer row", zap.Error(err))
			return nil, err
		}
		unflushedRowCount++

		if (unflushedRowCount+1)%100 == 0 && writer.IsFull() {
			serWriteStart := time.Now()
			kvs, partialBinlogs, err := serializeWrite(ctx, writer, t.Allocator)
			if err != nil {
				log.Warn("compact wrong, failed to serialize writer", zap.Error(err))
				return nil, err
			}
			serWriteTimeCost += time.Since(serWriteStart)

			uploadStart := time.Now()
			if err := t.binlogIO.Upload(ctx, kvs); err != nil {
				log.Warn("compact wrong, failed to upload kvs", zap.Error(err))
			}
			uploadTimeCost += time.Since(uploadStart)
			mergeFieldBinlogs(allBinlogs, partialBinlogs)
			syncBatchCount++
			unflushedRowCount = 0
		}
	}

	if !writer.IsEmpty() {
		serWriteStart := time.Now()
		kvs, partialBinlogs, err := serializeWrite(ctx, writer, t.Allocator)
		if err != nil {
			log.Warn("compact wrong, failed to serialize writer", zap.Error(err))
			return nil, err
		}
		serWriteTimeCost += time.Since(serWriteStart)

		uploadStart := time.Now()
		if err := t.binlogIO.Upload(ctx, kvs); err != nil {
			log.Warn("compact wrong, failed to upload kvs", zap.Error(err))
		}
		uploadTimeCost += time.Since(uploadStart)

		mergeFieldBinlogs(allBinlogs, partialBinlogs)
		syncBatchCount++
	}

	serWriteStart := time.Now()
	sPath, err := statSerializeWrite(ctx, writer, numRows, t.Allocator, t.binlogIO)
	if err != nil {
		log.Warn("compact wrong, failed to serialize write segment stats",
			zap.Int64("remaining row count", numRows), zap.Error(err))
		return nil, err
	}
	serWriteTimeCost += time.Since(serWriteStart)

	pack := &datapb.CompactionSegment{
		SegmentID:           writer.GetSegmentID(),
		InsertLogs:          lo.Values(allBinlogs),
		Field2StatslogPaths: []*datapb.FieldBinlog{sPath},
		NumOfRows:           int64(len(values)),
		Channel:             t.plan.GetChannel(),
		IsSorted:            true,
	}

	totalElapse := t.tr.RecordSpan()

	log.Info("sort segment end",
		zap.Int64("target segmentID", pack.GetSegmentID()),
		zap.Int64("old rows", numRows),
		zap.Int("valid rows", len(values)),
		zap.Int("binlog batch count", syncBatchCount),
		zap.Duration("upload binlogs elapse", uploadTimeCost),
		zap.Duration("sort elapse", sortTimeCost),
		zap.Duration("serWrite elapse", serWriteTimeCost),
		zap.Duration("total elapse", totalElapse),
		zap.Duration("compact elapse", time.Since(compactStart)))

	metrics.DataNodeCompactionLatency.WithLabelValues(fmt.Sprint(paramtable.GetNodeID()), t.plan.GetType().String()).Observe(float64(t.tr.ElapseSpan().Milliseconds()))
	metrics.DataNodeCompactionLatencyInQueue.WithLabelValues(fmt.Sprint(paramtable.GetNodeID())).Observe(float64(durInQueue.Milliseconds()))

	planResult := &datapb.CompactionPlanResult{
		State:    commonpb.CompactionState_Completed,
		PlanID:   t.GetPlanID(),
		Channel:  t.GetChannelName(),
		Segments: []*datapb.CompactionSegment{pack},
		Type:     t.plan.GetType(),
	}

	return planResult, nil
}

func (t *sortingCompactionTask) downloadData(ctx context.Context, numRows int64, PKFieldID int64) ([]*storage.Value, error) {
	log := log.With(zap.Int64("planID", t.GetPlanID()))
	if err := binlog.DecompressCompactionBinlogs(t.plan.GetSegmentBinlogs()); err != nil {
		log.Warn("sort segment wrong, fail to decompress compaction binlogs", zap.Error(err))
		return nil, err
	}

	allPath := make([][]string, 0)
	deltaPaths := make([]string, 0)
	binlogNum := len(t.plan.GetSegmentBinlogs()[0].GetFieldBinlogs()[0].GetBinlogs())
	for idx := 0; idx < binlogNum; idx++ {
		var batchPaths []string
		for _, f := range t.plan.GetSegmentBinlogs()[0].GetFieldBinlogs() {
			batchPaths = append(batchPaths, f.GetBinlogs()[idx].GetLogPath())
		}
		allPath = append(allPath, batchPaths)
	}
	for _, d := range t.plan.GetSegmentBinlogs()[0].GetDeltalogs() {
		for _, l := range d.GetBinlogs() {
			deltaPaths = append(deltaPaths, l.GetLogPath())
		}
	}

	deletePKs, err := t.loadDeltalogs(ctx, deltaPaths)
	if err != nil {
		log.Warn("load deletePKs failed", zap.Error(err))
		return nil, err
	}

	var (
		remainingRowCount int64 // the number of remaining entities
		expiredRowCount   int64 // the number of expired entities
	)

	isValueDeleted := func(v *storage.Value) bool {
		ts, ok := deletePKs[v.PK.GetValue()]
		// insert task and delete task has the same ts when upsert
		// here should be < instead of <=
		// to avoid the upsert data to be deleted after compact
		if ok && uint64(v.Timestamp) < ts {
			return true
		}
		return false
	}

	downloadTimeCost := time.Duration(0)

	values := make([]*storage.Value, 0, numRows)
	for _, paths := range allPath {
		log := log.With(zap.Strings("paths", paths))
		downloadStart := time.Now()
		allValues, err := t.binlogIO.Download(ctx, paths)
		if err != nil {
			log.Warn("compact wrong, fail to download insertLogs", zap.Error(err))
		}
		downloadTimeCost += time.Since(downloadStart)

		blobs := lo.Map(allValues, func(v []byte, i int) *storage.Blob {
			return &storage.Blob{Key: paths[i], Value: v}
		})

		iter, err := storage.NewBinlogDeserializeReader(blobs, PKFieldID)
		if err != nil {
			log.Warn("compact wrong, failed to new insert binlogs reader", zap.Error(err))
			return nil, err
		}

		for {
			err := iter.Next()
			if err != nil {
				if err == sio.EOF {
					break
				} else {
					log.Warn("compact wrong, failed to iter through data", zap.Error(err))
					return nil, err
				}
			}

			v := iter.Value()
			if isValueDeleted(v) {
				continue
			}

			// Filtering expired entity
			if t.isExpiredEntity(typeutil.Timestamp(v.Timestamp)) {
				expiredRowCount++
				continue
			}

			values = append(values, iter.Value())
			remainingRowCount++
		}
	}

	log.Info("download data success",
		zap.Int64("old rows", numRows),
		zap.Int64("remainingRowCount", remainingRowCount),
		zap.Int64("expiredRowCount", expiredRowCount),
		zap.Duration("download binlogs elapse", downloadTimeCost),
	)

	return values, nil
}

func (t *sortingCompactionTask) loadDeltalogs(ctx context.Context, dpaths []string) (map[interface{}]typeutil.Timestamp, error) {
	t.tr.RecordSpan()
	ctx, span := otel.Tracer(typeutil.DataNodeRole).Start(ctx, "loadDeltalogs")
	defer span.End()

	log := log.With(zap.Int64("planID", t.GetPlanID()))
	pk2ts := make(map[interface{}]typeutil.Timestamp)

	if len(dpaths) == 0 {
		log.Info("compact with no deltalogs, skip merge deltalogs")
		return pk2ts, nil
	}

	blobs, err := t.binlogIO.Download(ctx, dpaths)
	if err != nil {
		log.Warn("compact wrong, fail to download deltalogs", zap.Error(err))
		return nil, err
	}

	deltaIter := iter.NewDeltalogIterator(blobs, nil)
	for deltaIter.HasNext() {
		labeled, _ := deltaIter.Next()
		ts := labeled.GetTimestamp()
		if lastTs, ok := pk2ts[labeled.GetPk().GetValue()]; ok && lastTs > ts {
			ts = lastTs
		}
		pk2ts[labeled.GetPk().GetValue()] = ts
	}

	log.Info("compact loadDeltalogs end",
		zap.Int("deleted pk counts", len(pk2ts)),
		zap.Duration("elapse", t.tr.RecordSpan()))

	return pk2ts, nil
}

func (t *sortingCompactionTask) isExpiredEntity(ts typeutil.Timestamp) bool {
	now := t.currentTs

	// entity expire is not enabled if duration <= 0
	if t.plan.GetCollectionTtl() <= 0 {
		return false
	}

	entityT, _ := tsoutil.ParseTS(ts)
	nowT, _ := tsoutil.ParseTS(now)

	return entityT.Add(time.Duration(t.plan.GetCollectionTtl())).Before(nowT)
}
