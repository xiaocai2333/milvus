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

package importv2

import (
	"context"
	"sort"
	"sync"
	"time"

	"github.com/samber/lo"

	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/taskcommon"
	"github.com/milvus-io/milvus/pkg/v3/util/conc"
)

type Scheduler interface {
	Start()
	Slots() int64
	Resource() taskcommon.Resource
	Close()
}

type scheduler struct {
	manager TaskManager

	// futuresMu protects futures during submission and collection.
	futuresMu sync.Mutex
	// futures tracks in-flight file futures by task ID.
	futures map[int64][]*conc.Future[any]
	// draining keeps the booking of a failed task whose file futures are still
	// running. A failed task leaves the Pending/InProgress states at once, but
	// nothing cancels its sibling readers, so they keep their buffers until they
	// return. Protected by futuresMu.
	draining map[int64]drainingTask

	// submitMu allows only one goroutine to submit tasks at a time.
	submitMu sync.Mutex

	closeOnce sync.Once
	closeChan chan struct{}
}

// drainingTask is the booking of a failed task and the futures that still hold it.
type drainingTask struct {
	resource taskcommon.Resource
	futures  []*conc.Future[any]
}

func allDone(fs []*conc.Future[any]) bool {
	return lo.EveryBy(fs, func(f *conc.Future[any]) bool { return f.Done() })
}

func NewScheduler(manager TaskManager) Scheduler {
	return &scheduler{
		manager:   manager,
		futures:   make(map[int64][]*conc.Future[any]),
		draining:  make(map[int64]drainingTask),
		closeChan: make(chan struct{}),
	}
}

func (s *scheduler) Start() {
	mlog.Info(context.TODO(), "start import scheduler")

	var (
		exeTicker = time.NewTicker(1 * time.Second)
		logTicker = time.NewTicker(10 * time.Minute)
	)
	defer exeTicker.Stop()
	defer logTicker.Stop()

	for {
		select {
		case <-s.closeChan:
			mlog.Info(context.TODO(), "import scheduler exited")
			return
		case <-exeTicker.C:
			s.scheduleTasks()
			s.collectCompletedTasks()
		case <-logTicker.C:
			LogStats(s.manager)
		}
	}
}

func (s *scheduler) scheduleTasks() {
	if !s.submitMu.TryLock() {
		return
	}

	tasks := s.manager.GetBy(WithStates(datapb.ImportTaskStateV2_Pending))
	sort.Slice(tasks, func(i, j int) bool {
		return tasks[i].GetTaskID() < tasks[j].GetTaskID()
	})

	if len(tasks) == 0 {
		s.submitMu.Unlock()
		return
	}

	taskIDs := lo.Map(tasks, func(t Task, _ int) int64 {
		return t.GetTaskID()
	})
	mlog.Info(context.TODO(), "processing tasks...", mlog.Int64s("taskIDs", taskIDs))

	go func() {
		defer s.submitMu.Unlock()

		for _, task := range tasks {
			// Execute may block, so do not hold futuresMu while submitting.
			fs := task.Execute()

			s.futuresMu.Lock()
			s.futures[task.GetTaskID()] = fs
			s.futuresMu.Unlock()
		}
	}()
}

// collectCompletedTasks marks successful tasks Completed and removes failed or
// deleted tasks from tracking without waiting for unfinished futures. A failed
// task whose futures are still running keeps its booking in draining until
// they all return.
func (s *scheduler) collectCompletedTasks() {
	s.futuresMu.Lock()
	defer s.futuresMu.Unlock()

	for taskID, d := range s.draining {
		if allDone(d.futures) {
			delete(s.draining, taskID)
		}
	}

	for taskID, fs := range s.futures {
		task := s.manager.Get(taskID)
		if task == nil {
			delete(s.futures, taskID)
			continue
		}
		if task.GetState() == datapb.ImportTaskStateV2_Failed {
			delete(s.futures, taskID)
			if !allDone(fs) {
				s.draining[taskID] = drainingTask{resource: task.GetResource(), futures: fs}
			}
			mlog.Warn(context.TODO(), "preimport/import failed", WrapLogFields(task, mlog.String("reason", task.GetReason()))...)
			continue
		}
		if !allDone(fs) {
			continue
		}
		delete(s.futures, taskID)
		if err := conc.AwaitAll(fs...); err != nil {
			mlog.Warn(context.TODO(), "preimport/import failed", WrapLogFields(task, mlog.Err(err))...)
			continue
		}
		s.manager.Update(taskID, UpdateState(datapb.ImportTaskStateV2_Completed))
		mlog.Info(context.TODO(), "preimport/import done", mlog.FieldTaskID(taskID))
	}
}

// Slots returns the used slots for import
func (s *scheduler) Slots() int64 {
	tasks := s.manager.GetBy(WithStates(datapb.ImportTaskStateV2_Pending, datapb.ImportTaskStateV2_InProgress))
	used := lo.SumBy(tasks, func(t Task) int64 {
		return t.GetSlots()
	})
	return used
}

// Resource returns the cpu/memory booked by pending and in-progress import
// tasks, plus failed tasks whose file futures have not all returned yet.
//
// Memory is never reported below what the import memory allocator has
// reserved. DataCoord prices a file by min(buffer, file size), but every
// reader reserves its whole buffer from the allocator and a reader such as
// Parquet preallocates its batches by buffer capacity, not by the file. The
// allocator bounds that reservation to dataNode.import.memoryLimitPercentage of
// the machine, so this keeps the reported free memory honest without pricing
// every small file at a whole buffer.
func (s *scheduler) Resource() taskcommon.Resource {
	tasks := s.manager.GetBy(WithStates(datapb.ImportTaskStateV2_Pending, datapb.ImportTaskStateV2_InProgress))
	var used taskcommon.Resource
	for _, t := range tasks {
		used = used.Add(t.GetResource())
	}
	s.futuresMu.Lock()
	defer s.futuresMu.Unlock()
	for _, d := range s.draining {
		if !allDone(d.futures) {
			used = used.Add(d.resource)
		}
	}
	used.Memory = max(used.Memory, GetMemoryAllocator().Used())
	return used
}

func (s *scheduler) Close() {
	s.closeOnce.Do(func() {
		close(s.closeChan)
	})
}
