// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. See the NOTICE file distributed with this work for
// additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

package taskcost

import (
	"math"
	"sync"
)

// MemoryAdmission is the shared execution budget for index/stats/analyze and
// import V3. Waiting work stays in each scheduler's acceptance ledger; it must
// not reserve execution memory and prevent its own admission.
//
// externalUsage covers the other executors only. It must not read either
// participating scheduler: they call TryAcquire under their own locks.
// CPU and scalar slots never gate this budget.
type MemoryAdmission struct {
	mu            sync.Mutex
	used          int64
	capacity      func() int64
	externalUsage func() int64
}

// NewMemoryAdmission creates an execution budget, separate from queued bookings.
// Both callbacks are required; capacity may report zero when hardware is unknown.
func NewMemoryAdmission(capacity, externalUsage func() int64) *MemoryAdmission {
	return &MemoryAdmission{capacity: capacity, externalUsage: externalUsage}
}

// Capacity returns the current non-negative node memory budget in bytes.
func (b *MemoryAdmission) Capacity() int64 {
	return b.capacity()
}

// TryAcquire atomically reserves a supplied memory price and returns a release
// called once by the admitted execution. An otherwise idle node can execute one
// oversized estimate alone, matching coordinator placement. Unknown capacity
// retains legacy admission while tracking reservations for capacity recovery.
func (b *MemoryAdmission) TryAcquire(memory int64) (func(), bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	capacity := b.Capacity()
	external := b.externalUsage()
	if capacity > 0 && (b.used > 0 || external > 0) {
		// Subtract capped terms instead of adding usages, which could wrap
		// for an oversized price and incorrectly offer memory again.
		remaining := max(capacity-min(b.used, capacity)-min(external, capacity), 0)
		if memory > remaining {
			return nil, false
		}
	}
	if memory > math.MaxInt64-b.used {
		return nil, false
	}
	b.used += memory
	return func() {
		b.mu.Lock()
		b.used -= memory
		b.mu.Unlock()
	}, true
}
