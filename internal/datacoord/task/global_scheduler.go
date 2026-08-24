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

package task

import (
	"context"
	"sync"
	"time"

	"github.com/milvus-io/milvus/internal/datacoord/session"
	"github.com/milvus-io/milvus/internal/util/taskresource"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	taskcommon "github.com/milvus-io/milvus/pkg/v3/taskcommon"
	"github.com/milvus-io/milvus/pkg/v3/util/conc"
	"github.com/milvus-io/milvus/pkg/v3/util/lock"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

const NullNodeID = -1

type GlobalScheduler interface {
	Enqueue(task Task)
	AbortAndRemoveTask(taskID int64)
	// GetPendingTaskCount returns the number of queued tasks of the given type.
	// The queue is shared by every task type, so callers that gate admission for
	// one kind of work must scope the count to that kind, otherwise an unrelated
	// backlog starves them. Tasks waiting on a retry backoff deadline ARE counted:
	// they still occupy queue depth, and excluding them would let a worker-side
	// failure storm silently disable the caller's admission gate.
	GetPendingTaskCount(taskType taskcommon.Type) int

	Start()
	Stop()
}

var _ GlobalScheduler = (*globalTaskScheduler)(nil)

type globalTaskScheduler struct {
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	mu           *lock.KeyLock[int64]
	pendingTasks PriorityQueue
	runningTasks *typeutil.ConcurrentMap[int64, Task]
	execPool     *conc.Pool[struct{}]
	checkPool    *conc.Pool[struct{}]
	cluster      session.Cluster
	// backoffs delays re-dispatch of tasks that failed on a worker. Without
	// it a task that keeps failing (e.g. its object-storage reads are being
	// throttled) is re-sent every TaskScheduleInterval (~100ms), which turns
	// one bad task into a dispatch storm that keeps the store throttled.
	backoffs *typeutil.ConcurrentMap[int64, *taskBackoff]
}

// taskBackoff records how often a task failed on a worker and when it may be
// dispatched again. Entries are replaced wholesale (copy-on-write) so readers
// never observe a partially updated value.
type taskBackoff struct {
	failures  int
	notBefore time.Time
}

// recordTaskFailure schedules the next dispatch of a failed task with
// exponential backoff: interval * 2^(failures-1), capped at maxInterval.
func (s *globalTaskScheduler) recordTaskFailure(task Task) {
	interval := paramtable.Get().DataCoordCfg.TaskRetryBackoffInterval.GetAsDuration(time.Second)
	if interval <= 0 {
		return
	}
	maxInterval := paramtable.Get().DataCoordCfg.TaskRetryBackoffMaxInterval.GetAsDuration(time.Second)

	failures := 1
	if old, ok := s.backoffs.Get(task.GetTaskID()); ok {
		failures = old.failures + 1
	}
	// cap the shift to keep the doubling far away from overflow
	if shift := failures - 1; shift < 30 {
		interval <<= shift
	} else {
		interval = maxInterval
	}
	if maxInterval > 0 && interval > maxInterval {
		interval = maxInterval
	}
	s.backoffs.Insert(task.GetTaskID(), &taskBackoff{
		failures:  failures,
		notBefore: time.Now().Add(interval),
	})
	mlog.Info(s.ctx, "task failed on worker, backing off before retry",
		WrapTaskLog(task, mlog.Int("failures", failures), mlog.Duration("backoff", interval))...)
}

// taskInBackoff reports whether the task's next dispatch is still delayed.
func (s *globalTaskScheduler) taskInBackoff(task Task) bool {
	bo, ok := s.backoffs.Get(task.GetTaskID())
	return ok && time.Now().Before(bo.notBefore)
}

func (s *globalTaskScheduler) Enqueue(task Task) {
	if s.pendingTasks.Get(task.GetTaskID()) != nil {
		return
	}
	if s.runningTasks.Contain(task.GetTaskID()) {
		return
	}
	switch task.GetTaskState() {
	case taskcommon.Init:
		task.SetTaskTime(taskcommon.TimeQueue, time.Now())
		s.pendingTasks.Push(task)
	case taskcommon.InProgress, taskcommon.Retry:
		task.SetTaskTime(taskcommon.TimeStart, time.Now())
		s.runningTasks.Insert(task.GetTaskID(), task)
	}
	mlog.Info(s.ctx, "task enqueued", WrapTaskLog(task)...)
}

func (s *globalTaskScheduler) GetPendingTaskCount(taskType taskcommon.Type) int {
	return s.pendingTasks.TaskCountBy(func(task Task) bool {
		return task.GetTaskType() == taskType
	})
}

func (s *globalTaskScheduler) AbortAndRemoveTask(taskID int64) {
	s.mu.Lock(taskID)
	defer s.mu.Unlock(taskID)
	if task, ok := s.runningTasks.GetAndRemove(taskID); ok {
		task.DropTaskOnWorker(s.cluster)
	}
	if task := s.pendingTasks.Get(taskID); task != nil {
		task.DropTaskOnWorker(s.cluster)
		s.pendingTasks.Remove(taskID)
	}
	s.backoffs.Remove(taskID)
}

func (s *globalTaskScheduler) Start() {
	dur := paramtable.Get().DataCoordCfg.TaskScheduleInterval.GetAsDuration(time.Millisecond)
	s.wg.Add(3)
	go func() {
		defer s.wg.Done()
		t := time.NewTicker(dur)
		defer t.Stop()
		for {
			select {
			case <-s.ctx.Done():
				return
			case <-t.C:
				s.schedule()
			}
		}
	}()
	go func() {
		defer s.wg.Done()
		t := time.NewTicker(dur)
		defer t.Stop()
		for {
			select {
			case <-s.ctx.Done():
				return
			case <-t.C:
				s.check()
			}
		}
	}()
	go func() {
		defer s.wg.Done()
		t := time.NewTicker(time.Minute)
		defer t.Stop()
		for {
			select {
			case <-s.ctx.Done():
				return
			case <-t.C:
				s.updateTaskTimeMetrics()
			}
		}
	}()
}

func (s *globalTaskScheduler) Stop() {
	s.cancel()
	s.wg.Wait()
}

type nodeSlotEntry struct {
	nodeID int64
	slots  *session.WorkerSlots
}

// newNodeEntries flattens the query result into the slice both placement
// strategies work over. The entries are shared by pointer, so whichever strategy
// runs mutates the same state.
func newNodeEntries(workerSlots map[int64]*session.WorkerSlots) []*nodeSlotEntry {
	entries := make([]*nodeSlotEntry, 0, len(workerSlots))
	for nodeID, ws := range workerSlots {
		entries = append(entries, &nodeSlotEntry{nodeID: nodeID, slots: ws})
	}
	return entries
}

// allHaveDimensions reports whether every node reported CPU and memory, which is
// the condition for using two-dimensional placement this round.
//
// It requires ALL nodes rather than any, on purpose. During a rolling upgrade a
// mixed fleet would otherwise have the new nodes scored on two dimensions and
// the old ones excluded for having none, so every task would land on the
// upgraded half. Falling back to the scalar for the whole fleet until the last
// node restarts keeps the transition atomic and is the conservative direction:
// the scalar is what the cluster ran on yesterday.
func allHaveDimensions(entries []*nodeSlotEntry) bool {
	if len(entries) == 0 {
		return false
	}
	for _, e := range entries {
		if !e.slots.HasDimensions() {
			return false
		}
	}
	return true
}

// newNodeSlotHeap builds a max-heap of worker nodes ordered by their available
// slots, so the most-available (least-loaded) node always sits at the top.
func newNodeSlotHeap(workerSlots map[int64]*session.WorkerSlots) typeutil.Heap[*nodeSlotEntry] {
	return newNodeSlotHeapFromEntries(newNodeEntries(workerSlots))
}

func newNodeSlotHeapFromEntries(entries []*nodeSlotEntry) typeutil.Heap[*nodeSlotEntry] {
	return typeutil.NewObjectArrayBasedMaximumHeap(entries, func(entry *nodeSlotEntry) int64 {
		return entry.slots.AvailableSlots
	})
}

// scoreNode ranks a candidate for a task, in [0, 1]-ish units, using the two
// dimensions the node reports rather than the scalar fold of them.
//
// Both terms are FRACTIONS of that node's own capacity, not absolute remainders.
// Scoring on absolutes would make a bigger node win merely for being bigger,
// which is the opposite of load balancing on a heterogeneous fleet.
//
// The third term penalizes SKEW -- how far apart the two dimensions' utilisations
// would sit. Without it the scheduler happily drives a node to "CPU exhausted,
// half its memory stranded", and that stranded half can never be used again
// until something finishes. This is Kubernetes' BalancedAllocation, and it is
// the only reason reporting the totals was worth the proto change.
//
// What this deliberately does NOT do is filter on CPU. Memory is incompressible
// -- exceeding it kills the process -- while CPU merely slows things down, so
// only memory may ever exclude a node. Filtering on CPU here would recreate
// precisely the problem Requirement.MemoryFitsIn exists to avoid on the worker:
// an L0 compaction held behind vector index builds it shares no thread pool
// with. CPU therefore scores and never refuses.
func scoreNode(ws *session.WorkerSlots) float64 {
	memFrac, cpuFrac := 0.0, 0.0
	terms := 0.0
	score := 0.0

	if ws.MemoryTotal > 0 {
		memFrac = float64(ws.MemoryAvailable) / float64(ws.MemoryTotal)
		score += memFrac
		terms++
	}
	if ws.CPUTotalMilli > 0 {
		cpuFrac = float64(ws.CPUAvailableMilli) / float64(ws.CPUTotalMilli)
		score += cpuFrac
		terms++
	}
	if terms == 0 {
		return 0
	}
	score /= terms

	// Skew penalty only when both dimensions are known; with one dimension there
	// is nothing to strand.
	if ws.MemoryTotal > 0 && ws.CPUTotalMilli > 0 {
		skew := memFrac - cpuFrac
		if skew < 0 {
			skew = -skew
		}
		score -= balanceWeight * skew
	}
	return score
}

// balanceWeight is how much a skewed node is punished relative to a full one.
// At 0.5 a node whose two dimensions are one whole capacity apart loses half a
// dimension's worth of score -- enough to lose to a slightly fuller but even
// node, not enough to override a genuinely empty one.
const balanceWeight = 0.5

// pickNodeByDimensions places a task using the reported CPU and memory instead
// of the folded scalar. It returns NullNodeID when no node reported dimensions,
// so the caller can fall back to the scalar heap.
//
// Memory is a hard filter and CPU is not; see scoreNode. When nothing passes the
// memory filter the task is still dispatched, to the node with the most memory
// free, because the worker's guard admits an oversized task exclusively rather
// than refusing it -- refusing here would leave it pending forever, since no node
// ever grows.
//
// req is the task's own footprint in bytes and cores. A ZERO req means the task
// type has not been converted on the coordinator side yet: the filter then only
// asks whether the node has any memory left at all, which is the same
// information the scalar carried. A NON-ZERO req is what makes this an actual
// reservation -- the node's remainders are decremented by it, so the second
// task of a round sees the bytes the first one took, and a node cannot be handed
// ten 8GiB compactions because each of them individually fitted.
//
// The picked node's figures are decremented in place; the caller reuses the same
// entries across all tasks in a round.
func pickNodeByDimensions(entries []*nodeSlotEntry, taskSlot int64, req taskresource.Requirement) int64 {
	var best *nodeSlotEntry
	var bestScore float64
	var fallback *nodeSlotEntry
	var fallbackMem int64
	haveFallback := false

	for _, e := range entries {
		if !e.slots.HasDimensions() {
			continue
		}
		if !haveFallback || e.slots.MemoryAvailable > fallbackMem {
			fallback, fallbackMem, haveFallback = e, e.slots.MemoryAvailable, true
		}
		if !memoryFits(e.slots, req.Memory) {
			continue
		}
		if sc := scoreNode(e.slots); best == nil || sc > bestScore {
			best, bestScore = e, sc
		}
	}

	if best == nil {
		if !haveFallback {
			return NullNodeID // no node reported dimensions; caller falls back
		}
		best = fallback
	}

	chargeNode(best.slots, taskSlot, req)
	return best.nodeID
}

// memoryFits is the hard filter. With a known requirement it asks whether the
// node can hold it; without one it can only ask whether the node has anything
// left, which is what the scalar already told us.
//
// A node reporting no memory capacity at all is not filtered out: that is a
// node whose guard has not established a budget yet, and excluding it would
// take it out of service on the strength of a missing number.
func memoryFits(ws *session.WorkerSlots, memoryRequired int64) bool {
	if ws.MemoryTotal <= 0 {
		return true
	}
	if memoryRequired > 0 {
		return ws.MemoryAvailable >= memoryRequired
	}
	return ws.MemoryAvailable > 0
}

// chargeNode debits a placement from the node's view for the rest of this round.
//
// The scalar is debited too, so a mixed fleet and the next round both see the
// load. The dimensions are debited only when the requirement is known -- charging
// a zero requirement would leave the node looking untouched and let the whole
// round pile onto it.
//
// Remainders are allowed to go negative. That is the same choice the DataNode
// makes when it reports them: over-commitment is a fact the filter needs to see,
// and clamping it at zero would read as "exactly full" instead.
func chargeNode(ws *session.WorkerSlots, taskSlot int64, req taskresource.Requirement) {
	if taskSlot > 0 {
		if ws.AvailableSlots >= taskSlot {
			ws.AvailableSlots -= taskSlot
		} else {
			ws.AvailableSlots = 0
		}
	}
	if req.Memory > 0 {
		ws.MemoryAvailable -= req.Memory
	}
	if req.CPU > 0 {
		ws.CPUAvailableMilli -= int64(req.CPU * 1000)
	}
}

// pickNode selects the least-loaded node -- the one with the most available
// slots -- for a task requiring taskSlot slots. It is pickNodeWithMinimumVersion
// with no version constraint, so every node in the heap is a candidate.
//
// When no node can fully satisfy taskSlot it still dispatches, to that
// most-available node, and drains its slots. That is deliberate and not the
// defect issue #52180 named: a task larger than any worker has to run
// somewhere, and the emptiest worker is where it will start soonest. A worker
// running the exclusive-admission guard (see internal/datanode/resource)
// admits such a task only once every other reservation has been released and
// runs it alone, so the harm in #52180 -- an oversized task running
// *concurrently* with everything else -- cannot recur once every DataNode
// carries that guard. Refusing to place the task here would instead leave it
// pending forever, because no node ever grows.
//
// That precondition is not enforced on this path, and the worker-version
// filter in pickNodeWithMinimumVersion does not change it. The filter screens
// against a minimum version the *task* declares (Task.MinimumWorkerVersion),
// and the only task that declares one today is an external-snapshot copy
// segment; every oversized compaction, index or stats task reaches the heap
// with an empty constraint, which workerSupportsMinimumVersion accepts from
// every worker. Even a constrained task would not be a reliable screen for the
// guard, since an unparsable development version is treated as compatible. So
// during a partial rollout, a rollback, or before a node has restarted into
// the new build, an oversized task dispatched to a guard-less worker still
// behaves as it did in #52180.
//
// It returns NullNodeID when the heap holds no nodes at all, or when a task
// needing slots finds every node reporting none; either way the task waits for
// the next scheduling round. A task asking for no slots is placed on the
// most-available node without consuming any.
//
// The picked node's slots are updated in place; the caller reuses the same heap
// across all tasks in a scheduling round so later picks observe the decremented
// slots.
func (s *globalTaskScheduler) pickNode(slotHeap typeutil.Heap[*nodeSlotEntry], taskSlot int64) int64 {
	if slotHeap.Len() == 0 {
		return NullNodeID
	}
	// Pop the most-available node, mutate its slots, then push it back. An element
	// must not be mutated while it stays in the heap, or the heap order breaks.
	entry := slotHeap.Pop()
	if taskSlot <= 0 {
		slotHeap.Push(entry)
		return entry.nodeID
	}
	if entry.slots.AvailableSlots <= 0 {
		// The most-available node has no slot, so neither does any other node.
		slotHeap.Push(entry)
		return NullNodeID
	}
	if entry.slots.AvailableSlots >= taskSlot {
		entry.slots.AvailableSlots -= taskSlot
	} else {
		// No node can fully satisfy the request; assign to the most-available
		// node on a best-effort basis and drain its slots.
		entry.slots.AvailableSlots = 0
	}
	slotHeap.Push(entry)
	return entry.nodeID
}

func (s *globalTaskScheduler) schedule() {
	pendingNum := s.pendingTasks.TaskCount()
	if pendingNum == 0 {
		return
	}
	nodeSlots := s.cluster.QuerySlot()
	mlog.Info(s.ctx, "scheduling pending tasks...", mlog.Int("num", pendingNum), mlog.Any("nodeSlots", nodeSlots))

	// Build the node view once per round and reuse it across all picks, so each
	// task is placed on the currently least-loaded node.
	//
	// Only one of the two structures is used per round. They share entry
	// pointers, so mutating an entry through the slice would silently break the
	// heap's ordering invariant -- picking the strategy up front instead of per
	// task is what keeps that from happening.
	entries := newNodeEntries(nodeSlots)
	useDimensions := allHaveDimensions(entries)
	var slotHeap typeutil.Heap[*nodeSlotEntry]
	if !useDimensions {
		slotHeap = newNodeSlotHeapFromEntries(entries)
	}
	futures := make([]*conc.Future[struct{}], 0)
	var delayed []Task
	for {
		task := s.pendingTasks.Pop()
		if task == nil {
			break
		}
		// A task in failure backoff gives way: it re-enters the queue after
		// this round and is dispatched once its delay elapses, so one
		// persistently failing task cannot occupy the scheduler.
		if s.taskInBackoff(task) {
			delayed = append(delayed, task)
			continue
		}
		taskSlot := task.GetTaskSlot()
		var nodeID int64
		if useDimensions {
			nodeID = pickNodeByDimensions(entries, taskSlot, task.GetResourceRequirement())
		} else {
			nodeID = s.pickNode(slotHeap, taskSlot)
		}
		if nodeID == NullNodeID {
			s.pendingTasks.Push(task)
			break
		}
		future := s.execPool.Submit(func() (struct{}, error) {
			s.mu.RLock(task.GetTaskID())
			defer s.mu.RUnlock(task.GetTaskID())
			mlog.Info(s.ctx, "processing task...", WrapTaskLog(task)...)
			if task.GetTaskState() == taskcommon.Init {
				task.CreateTaskOnWorker(nodeID, s.cluster)
				switch task.GetTaskState() {
				case taskcommon.Init, taskcommon.Retry:
					s.recordTaskFailure(task)
					s.pendingTasks.Push(task)
				case taskcommon.InProgress:
					// The task was accepted by the worker and is now in flight.
					// Any accumulated failure count is intentionally kept: reaching
					// InProgress only means a slot happened to be free, not that the
					// cause of earlier failures is gone. If the task fails again the
					// backoff must keep escalating rather than restart from scratch.
					// The entry is cleared only on a terminal state (here and in
					// check()).
					task.SetTaskTime(taskcommon.TimeStart, time.Now())
					s.runningTasks.Insert(task.GetTaskID(), task)
				case taskcommon.None, taskcommon.Finished, taskcommon.Failed:
					// CreateTaskOnWorker can drive a task straight to a terminal
					// state (e.g. missing meta, unhealthy segment, estimation
					// failure). Such a task leaves the scheduler without ever
					// entering runningTasks, so check()'s terminal-state cleanup
					// never runs. Drop the backoff entry here; otherwise it would
					// leak until datacoord restarts and grow without bound under
					// the very failure storms this backoff exists to relieve.
					s.backoffs.Remove(task.GetTaskID())
				}
			}
			return struct{}{}, nil
		})
		futures = append(futures, future)
	}
	for _, task := range delayed {
		s.pendingTasks.Push(task)
	}
	_ = conc.AwaitAll(futures...)
}

func (s *globalTaskScheduler) check() {
	if s.runningTasks.Len() <= 0 {
		return
	}
	mlog.Info(s.ctx, "check running tasks", mlog.Int("num", s.runningTasks.Len()))

	tasks := s.runningTasks.Values()
	futures := make([]*conc.Future[struct{}], 0, len(tasks))
	for _, task := range tasks {
		future := s.checkPool.Submit(func() (struct{}, error) {
			s.mu.RLock(task.GetTaskID())
			defer s.mu.RUnlock(task.GetTaskID())
			task.QueryTaskOnWorker(s.cluster)
			switch task.GetTaskState() {
			case taskcommon.None:
				s.runningTasks.Remove(task.GetTaskID())
				s.backoffs.Remove(task.GetTaskID())
			case taskcommon.Init, taskcommon.Retry:
				s.recordTaskFailure(task)
				s.runningTasks.Remove(task.GetTaskID())
				s.pendingTasks.Push(task)
			case taskcommon.Finished, taskcommon.Failed:
				task.SetTaskTime(taskcommon.TimeEnd, time.Now())
				task.DropTaskOnWorker(s.cluster)
				s.runningTasks.Remove(task.GetTaskID())
				s.backoffs.Remove(task.GetTaskID())
			}
			return struct{}{}, nil
		})
		futures = append(futures, future)
	}
	_ = conc.AwaitAll(futures...)
}

func (s *globalTaskScheduler) updateTaskTimeMetrics() {
	var (
		taskNumByTypeAndState = make(map[string]map[string]int64) // taskType => [taskState => taskNum]
		maxTaskQueueingTime   = make(map[string]int64)
		maxTaskRunningTime    = make(map[string]int64)
	)

	for _, taskType := range taskcommon.TypeList {
		taskNumByTypeAndState[taskType] = make(map[string]int64)
	}

	collectPendingMetricsFunc := func(taskID int64) {
		task := s.pendingTasks.Get(taskID)
		if task == nil {
			return
		}

		s.mu.Lock(taskID)
		defer s.mu.Unlock(taskID)

		taskType := task.GetTaskType()

		queueingTime := time.Since(task.GetTaskTime(taskcommon.TimeQueue))
		if queueingTime > paramtable.Get().DataCoordCfg.TaskSlowThreshold.GetAsDuration(time.Second) {
			mlog.Warn(s.ctx, "task queueing time is too long", mlog.FieldTaskID(taskID),
				mlog.Int64("queueing time(ms)", queueingTime.Milliseconds()))
		}

		maxQueueingTime, ok := maxTaskQueueingTime[taskType]
		if !ok || maxQueueingTime < queueingTime.Milliseconds() {
			maxTaskQueueingTime[taskType] = queueingTime.Milliseconds()
		}

		taskNumByTypeAndState[taskType][task.GetTaskState().String()]++
		metrics.TaskVersion.WithLabelValues(taskType).Observe(float64(task.GetTaskVersion()))
	}

	collectRunningMetricsFunc := func(task Task) {
		s.mu.Lock(task.GetTaskID())
		defer s.mu.Unlock(task.GetTaskID())

		taskType := task.GetTaskType()

		runningTime := time.Since(task.GetTaskTime(taskcommon.TimeStart))
		if runningTime > paramtable.Get().DataCoordCfg.TaskSlowThreshold.GetAsDuration(time.Second) {
			mlog.Warn(s.ctx, "task running time is too long", mlog.FieldTaskID(task.GetTaskID()),
				mlog.Int64("running time(ms)", runningTime.Milliseconds()))
		}

		maxRunningTime, ok := maxTaskRunningTime[taskType]
		if !ok || maxRunningTime < runningTime.Milliseconds() {
			maxTaskRunningTime[taskType] = runningTime.Milliseconds()
		}

		taskNumByTypeAndState[taskType][task.GetTaskState().String()]++
	}

	taskIDs := s.pendingTasks.TaskIDs()

	for _, taskID := range taskIDs {
		collectPendingMetricsFunc(taskID)
	}

	allRunningTasks := s.runningTasks.Values()
	for _, task := range allRunningTasks {
		collectRunningMetricsFunc(task)
	}

	for taskType, queueingTime := range maxTaskQueueingTime {
		metrics.DataCoordTaskExecuteLatency.
			WithLabelValues(taskType, metrics.Pending).Observe(float64(queueingTime))
	}

	for taskType, runningTime := range maxTaskRunningTime {
		metrics.DataCoordTaskExecuteLatency.
			WithLabelValues(taskType, metrics.Executing).Observe(float64(runningTime))
	}

	metrics.TaskNumInGlobalScheduler.Reset()
	for taskType, taskNumByState := range taskNumByTypeAndState {
		for taskState, taskNum := range taskNumByState {
			metrics.TaskNumInGlobalScheduler.WithLabelValues(taskType, taskState).Set(float64(taskNum))
		}
	}
}

func NewGlobalTaskScheduler(ctx context.Context, cluster session.Cluster) GlobalScheduler {
	execPool := conc.NewPool[struct{}](128)
	checkPool := conc.NewPool[struct{}](128)
	ctx1, cancel := context.WithCancel(ctx)
	return &globalTaskScheduler{
		ctx:          ctx1,
		cancel:       cancel,
		wg:           sync.WaitGroup{},
		mu:           lock.NewKeyLock[int64](),
		pendingTasks: NewPriorityQueuePolicy(),
		runningTasks: typeutil.NewConcurrentMap[int64, Task](),
		execPool:     execPool,
		checkPool:    checkPool,
		cluster:      cluster,
		backoffs:     typeutil.NewConcurrentMap[int64, *taskBackoff](),
	}
}
