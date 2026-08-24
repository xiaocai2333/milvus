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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/datacoord/session"
	"github.com/milvus-io/milvus/internal/util/taskresource"
)

const (
	gib   = int64(1) << 30
	cores = int64(1000) // one core in millicores
)

func node(id, cpuAvail, cpuTotal, memAvail, memTotal, slots int64) *nodeSlotEntry {
	return &nodeSlotEntry{
		nodeID: id,
		slots: &session.WorkerSlots{
			NodeID:            id,
			AvailableSlots:    slots,
			CPUAvailableMilli: cpuAvail,
			MemoryAvailable:   memAvail,
			CPUTotalMilli:     cpuTotal,
			MemoryTotal:       memTotal,
		},
	}
}

// A node that never reported the new fields must be distinguishable from a node
// that reported them and is simply full. Zero *available* is ordinary; zero
// *totals* is what marks an old node.
func TestHasDimensionsDistinguishesOldNodeFromFullNode(t *testing.T) {
	old := &session.WorkerSlots{NodeID: 1, AvailableSlots: 64}
	assert.False(t, old.HasDimensions())

	full := &session.WorkerSlots{NodeID: 2, CPUTotalMilli: 8 * cores, MemoryTotal: 32 * gib}
	assert.True(t, full.HasDimensions(), "a full node still reports its capacity")

	var nilWS *session.WorkerSlots
	assert.False(t, nilWS.HasDimensions())
}

// Two dimensions are used only when EVERY node reports them. A mixed fleet mid
// upgrade must stay on the scalar, or all work lands on the upgraded half.
func TestAllHaveDimensionsRequiresEveryNode(t *testing.T) {
	newNode := node(1, 8*cores, 8*cores, 32*gib, 32*gib, 64)
	oldNode := node(2, 0, 0, 0, 0, 64)

	assert.True(t, allHaveDimensions([]*nodeSlotEntry{newNode}))
	assert.False(t, allHaveDimensions([]*nodeSlotEntry{newNode, oldNode}),
		"one old node must send the whole round back to the scalar")
	assert.False(t, allHaveDimensions(nil))
}

// The defining property: placement follows the dimensions, not the scalar. Node
// 2 advertises far more legacy slots but is nearly out of memory; node 1 has the
// memory. A scalar picker would choose 2.
func TestPickByDimensionsIgnoresTheScalar(t *testing.T) {
	n1 := node(1, 4*cores, 8*cores, 30*gib, 32*gib, 1)
	n2 := node(2, 8*cores, 8*cores, 1*gib, 32*gib, 1000)

	assert.Equal(t, int64(1), pickNodeByDimensions([]*nodeSlotEntry{n1, n2}, 0, taskresource.Requirement{}))
}

// Memory is a hard filter: a node with none left is not a candidate, however
// much CPU it still has.
func TestPickByDimensionsFiltersOnMemoryOnly(t *testing.T) {
	drained := node(1, 8*cores, 8*cores, 0, 32*gib, 100)
	viable := node(2, 1*cores, 8*cores, 8*gib, 32*gib, 1)

	assert.Equal(t, int64(2), pickNodeByDimensions([]*nodeSlotEntry{drained, viable}, 0, taskresource.Requirement{}))
}

// CPU is NOT a filter. A node whose CPU is fully committed must still be able to
// take memory-bound work -- this is the L0-compaction-behind-vector-builds case
// the worker's MemoryFitsIn exists to avoid, and filtering here would recreate it.
func TestPickByDimensionsNeverFiltersOnCPU(t *testing.T) {
	cpuExhausted := node(1, 0, 8*cores, 30*gib, 32*gib, 10)

	got := pickNodeByDimensions([]*nodeSlotEntry{cpuExhausted}, 0, taskresource.Requirement{})
	assert.Equal(t, int64(1), got, "an exhausted CPU dimension must not exclude a node")
}

// When nothing passes the memory filter the task is still dispatched, to the
// node with the most memory free. The worker's guard runs an oversized task
// exclusively; refusing here would leave it pending forever because no node
// ever grows.
func TestPickByDimensionsDispatchesWhenNothingFitsAnywhere(t *testing.T) {
	a := node(1, 8*cores, 8*cores, 0, 32*gib, 0)
	b := node(2, 8*cores, 8*cores, -1*gib, 32*gib, 0) // over-committed: negative is the truth

	got := pickNodeByDimensions([]*nodeSlotEntry{a, b}, 0, taskresource.Requirement{})
	assert.Equal(t, int64(1), got, "the least-bad node still receives the task")
}

// Balance: between two nodes with the same average headroom, the even one wins,
// so the scheduler does not drive a node to "CPU gone, memory stranded".
func TestScorePenalisesSkew(t *testing.T) {
	even := node(1, 4*cores, 8*cores, 16*gib, 32*gib, 0)   // 50% / 50%
	skewed := node(2, 1*cores, 8*cores, 28*gib, 32*gib, 0) // 12.5% / 87.5%

	evenScore := scoreNode(even.slots)
	skewedScore := scoreNode(skewed.slots)
	require.InDelta(t, 0.5, (float64(even.slots.CPUAvailableMilli)/float64(even.slots.CPUTotalMilli)+
		float64(even.slots.MemoryAvailable)/float64(even.slots.MemoryTotal))/2, 1e-9,
		"setup: both nodes must have the same mean headroom for this to test skew alone")

	assert.Greater(t, evenScore, skewedScore)
	assert.Equal(t, int64(1), pickNodeByDimensions([]*nodeSlotEntry{even, skewed}, 0, taskresource.Requirement{}))
}

// Fractions, not absolute remainders: a big node must not win merely for being
// big. Node 2 has more free bytes but a smaller share of itself free.
func TestScoreUsesFractionsNotAbsolutes(t *testing.T) {
	small := node(1, 4*cores, 4*cores, 7*gib, 8*gib, 0)     // 100% / 87.5%
	large := node(2, 8*cores, 64*cores, 20*gib, 256*gib, 0) // 12.5% / 7.8%

	assert.Greater(t, large.slots.MemoryAvailable, small.slots.MemoryAvailable,
		"setup: the big node must have more absolute bytes free")
	assert.Equal(t, int64(1), pickNodeByDimensions([]*nodeSlotEntry{small, large}, 0, taskresource.Requirement{}))
}

// The scalar is kept in step so a later round, or a mixed fleet, still sees the
// load this placement added.
func TestPickByDimensionsDecrementsTheScalar(t *testing.T) {
	n := node(1, 8*cores, 8*cores, 32*gib, 32*gib, 10)
	require.Equal(t, int64(1), pickNodeByDimensions([]*nodeSlotEntry{n}, 4, taskresource.Requirement{}))
	assert.Equal(t, int64(6), n.slots.AvailableSlots)

	// Over-asking drains rather than going negative.
	require.Equal(t, int64(1), pickNodeByDimensions([]*nodeSlotEntry{n}, 100, taskresource.Requirement{}))
	assert.Equal(t, int64(0), n.slots.AvailableSlots)
}

// No node reported dimensions: the caller must be told to fall back rather than
// be handed an arbitrary node.
func TestPickByDimensionsReturnsNullWhenNoNodeReports(t *testing.T) {
	a := node(1, 0, 0, 0, 0, 50)
	b := node(2, 0, 0, 0, 0, 90)

	assert.Equal(t, int64(NullNodeID), pickNodeByDimensions([]*nodeSlotEntry{a, b}, 1, taskresource.Requirement{}))
	assert.Equal(t, int64(50), a.slots.AvailableSlots, "a fallback round must not consume slots")
	assert.Equal(t, int64(90), b.slots.AvailableSlots)
}

// One dimension known is still usable, and must not be dragged down by a skew
// term computed against a capacity of zero.
func TestScoreWithOnlyOneDimension(t *testing.T) {
	memOnly := node(1, 0, 0, 16*gib, 32*gib, 0)
	assert.InDelta(t, 0.5, scoreNode(memOnly.slots), 1e-9)

	cpuOnly := node(2, 2*cores, 8*cores, 0, 0, 0)
	assert.InDelta(t, 0.25, scoreNode(cpuOnly.slots), 1e-9)

	none := node(3, 0, 0, 0, 0, 0)
	assert.Equal(t, 0.0, scoreNode(none.slots))
}

// The point of carrying the requirement: a node must not be handed several tasks
// that each fitted individually. Two 8GiB compactions onto a node with 12GiB free
// means the second one has to go elsewhere.
func TestRequirementIsReservedAcrossPicksInARound(t *testing.T) {
	big := taskresource.Requirement{CPU: 1, Memory: 8 * gib}
	a := node(1, 8*cores, 8*cores, 12*gib, 32*gib, 100)
	b := node(2, 8*cores, 8*cores, 10*gib, 32*gib, 100)
	entries := []*nodeSlotEntry{a, b}

	first := pickNodeByDimensions(entries, 1, big)
	second := pickNodeByDimensions(entries, 1, big)

	assert.Equal(t, int64(1), first, "the roomier node goes first")
	assert.Equal(t, int64(2), second,
		"the first placement must have consumed node 1's headroom, pushing the second elsewhere")
	assert.Equal(t, 4*gib, a.slots.MemoryAvailable)
	assert.Equal(t, 2*gib, b.slots.MemoryAvailable)
	assert.Equal(t, 7*cores, a.slots.CPUAvailableMilli, "CPU is debited too, in millicores")
}

// A known requirement filters on whether it FITS, not merely on whether anything
// is left. A node with 1GiB free is not a candidate for an 8GiB task.
func TestKnownRequirementFiltersOnFit(t *testing.T) {
	tight := node(1, 8*cores, 8*cores, 1*gib, 32*gib, 100)
	roomy := node(2, 1*cores, 8*cores, 20*gib, 32*gib, 1)

	got := pickNodeByDimensions([]*nodeSlotEntry{tight, roomy}, 1,
		taskresource.Requirement{Memory: 8 * gib})
	assert.Equal(t, int64(2), got)
}

// Nothing fits anywhere: still dispatched, to the roomiest node, because the
// worker runs an oversized task exclusively rather than refusing it.
func TestOversizedRequirementStillDispatches(t *testing.T) {
	a := node(1, 8*cores, 8*cores, 2*gib, 32*gib, 0)
	b := node(2, 8*cores, 8*cores, 5*gib, 32*gib, 0)

	got := pickNodeByDimensions([]*nodeSlotEntry{a, b}, 0,
		taskresource.Requirement{Memory: 100 * gib})
	assert.Equal(t, int64(2), got, "the emptiest node takes it and the worker serialises it")
}

// An unconverted task type reports a zero requirement. That must not read as
// "needs nothing": the node's dimensions stay untouched, so the round cannot
// pile onto one node on the strength of a missing number.
func TestZeroRequirementDoesNotDebitDimensions(t *testing.T) {
	n := node(1, 8*cores, 8*cores, 16*gib, 32*gib, 10)

	got := pickNodeByDimensions([]*nodeSlotEntry{n}, 2, taskresource.Requirement{})
	require.Equal(t, int64(1), got)
	assert.Equal(t, 16*gib, n.slots.MemoryAvailable, "no requirement, no debit")
	assert.Equal(t, 8*cores, n.slots.CPUAvailableMilli)
	assert.Equal(t, int64(8), n.slots.AvailableSlots, "the scalar is still debited")
}

// A node whose guard has not established a budget yet reports no capacity. It
// must stay in service rather than be excluded on a missing number.
func TestNodeWithoutMemoryCapacityIsNotFiltered(t *testing.T) {
	ws := &session.WorkerSlots{NodeID: 1, CPUTotalMilli: 8 * cores, CPUAvailableMilli: 4 * cores}
	assert.True(t, memoryFits(ws, 8*gib))
	assert.True(t, memoryFits(ws, 0))
}

// Over-commitment is allowed to go negative, matching how the DataNode reports
// it: the filter must read "does not fit", not "exactly full".
func TestChargeNodeAllowsNegativeRemainder(t *testing.T) {
	ws := &session.WorkerSlots{
		NodeID: 1, AvailableSlots: 1,
		MemoryTotal: 32 * gib, MemoryAvailable: 1 * gib,
		CPUTotalMilli: 8 * cores, CPUAvailableMilli: 500,
	}
	chargeNode(ws, 5, taskresource.Requirement{CPU: 2, Memory: 4 * gib})

	assert.Equal(t, -3*gib, ws.MemoryAvailable)
	assert.Equal(t, int64(-1500), ws.CPUAvailableMilli)
	assert.Equal(t, int64(0), ws.AvailableSlots, "the scalar clamps, as it always has")
	assert.False(t, memoryFits(ws, 1), "a negative remainder must fail the filter")
}
