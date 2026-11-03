package router

import (
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/process"
)

// TestBudget_AdmissionEvictsForDeclaredSize checks that a target with a
// declared sizeMB makes room before it loads, even though current usage alone
// is under the limit.
func TestBudget_AdmissionEvictsForDeclaredSize(t *testing.T) {
	a := newFakeProcess("a")
	a.markReady()
	b := newFakeProcess("b")
	b.markReady()
	processes := map[string]process.Process{"a": a, "b": b, "c": newFakeProcess("c")}
	s := newTestBudgetSwapper(10000, map[string]int{"a": 1, "b": 5}, processes, 8000)
	s.declared = map[string]int64{"a": 3000, "b": 5000, "c": 4000}

	// 8000 + 4000 > 10000: evicting a (3000) leaves 9000.
	evict := s.EvictionFor("c", []string{"a", "b"})
	if !slices.Equal(evict, []string{"a"}) {
		t.Fatalf("evict=%v want [a]", evict)
	}

	// Without a declared size for c, 8000 fits and nothing is evicted.
	s.declared = map[string]int64{"a": 3000, "b": 5000}
	if evict := s.EvictionFor("c", []string{"a", "b"}); len(evict) != 0 {
		t.Fatalf("evict=%v want none (target estimate unknown, usage fits)", evict)
	}
}

// TestBudget_AdmissionEvictsByEstimateNotShare checks that evicting a small
// model frees its own estimate, not an equal share of the total.
func TestBudget_AdmissionEvictsByEstimateNotShare(t *testing.T) {
	small := newFakeProcess("small")
	small.markReady()
	big := newFakeProcess("big")
	big.markReady()
	processes := map[string]process.Process{"small": small, "big": big, "c": newFakeProcess("c")}
	// small has the lower priority, so it goes first. An equal share would
	// say evicting it frees 5000 and stop there; it really frees 1000.
	s := newTestBudgetSwapper(10000, map[string]int{"small": 1, "big": 5}, processes, 10000)
	s.declared = map[string]int64{"small": 1000, "big": 9000, "c": 2000}

	evict := s.EvictionFor("c", []string{"small", "big"})
	if !slices.Equal(evict, []string{"small", "big"}) {
		t.Fatalf("evict=%v want [small big]", evict)
	}
}

// TestBudget_PickEvictionsMixedEstimates checks the share for models with no
// estimate: what the known estimates of resident models leave unexplained,
// split across the unknown models.
func TestBudget_PickEvictionsMixedEstimates(t *testing.T) {
	known := newFakeProcess("known")
	known.markReady()
	u1 := newFakeProcess("u1")
	u1.markReady()
	u2 := newFakeProcess("u2")
	u2.markReady()
	processes := map[string]process.Process{"known": known, "u1": u1, "u2": u2}
	s := newTestBudgetSwapper(5000, map[string]int{"u1": 1, "u2": 2, "known": 3}, processes, 0)
	s.declared = map[string]int64{"known": 6000}
	pool := []string{"known", "u1", "u2"}

	// sampled 10000: known accounts for 6000, so each unknown frees
	// (10000-6000)/2 = 2000. Evicting u1 leaves 8000, u2 leaves 6000, then
	// known (6000) leaves 0.
	evict := s.pickEvictions(pool, pool, 10000, 0)
	if !slices.Equal(evict, []string{"u1", "u2", "known"}) {
		t.Fatalf("evict=%v want [u1 u2 known]", evict)
	}

	// sampled 7000 with limit 5000: u1 frees 500 ((7000-6000)/2), u2 another
	// 500, known 6000.
	evict = s.pickEvictions(pool, pool, 7000, 0)
	if !slices.Equal(evict, []string{"u1", "u2", "known"}) {
		t.Fatalf("evict=%v want [u1 u2 known]", evict)
	}

	// Known estimates exceeding the sample clamp the share at 0 rather than
	// going negative.
	s.declared = map[string]int64{"known": 20000}
	evict = s.pickEvictions([]string{"u1", "u2", "known"}, pool, 10000, 0)
	if !slices.Equal(evict, []string{"u1", "u2", "known"}) {
		t.Fatalf("evict=%v want [u1 u2 known] (unknown share clamped at 0)", evict)
	}
}

// TestBudget_ReservationUsesDeclaredSize checks that a loading model with a
// declared size is reserved for before it has any learned history.
func TestBudget_ReservationUsesDeclaredSize(t *testing.T) {
	loading := newFakeProcess("loading")
	loading.setState(process.StateStarting)
	ready := newFakeProcess("ready")
	ready.markReady()
	processes := map[string]process.Process{"loading": loading, "ready": ready, "c": newFakeProcess("c")}
	s := newTestBudgetSwapper(10000, map[string]int{"ready": 1, "loading": 5}, processes, 5000)
	s.declared = map[string]int64{"loading": 6000}

	if got := s.reservedForInFlightMB([]string{"loading", "ready"}); got != 6000 {
		t.Fatalf("reserved=%d want 6000", got)
	}
	// 5000 + 6000 reserved > 10000, so the ready model goes.
	if evict := s.EvictionFor("c", []string{"loading", "ready"}); !slices.Equal(evict, []string{"ready"}) {
		t.Fatalf("evict=%v want [ready]", evict)
	}
}

// TestBudget_PerProcessEstimateOverridesLearned checks that a per-process
// VRAM figure, matched through the model's process tree, becomes the model's
// estimate over its learned median, is recorded into learned once per load,
// and survives the model stopping.
func TestBudget_PerProcessEstimateOverridesLearned(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 100000}}}},
	}
	var vramMB atomic.Int64
	vramMB.Store(1000)
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a}, &vramMB)
	r.swapper.recordLearned([]string{"a"}, 9000)
	r.swapper.recordLearned([]string{"a"}, 9000)
	r.swapper.recordLearned([]string{"a"}, 9000)

	// llama-server is a child (pid 101) of the wrapper llama-swap started
	// (pid 100); nvidia-smi reports only the child.
	r.swapper.descendants = func(pid int) map[int]bool {
		if pid == 100 {
			return map[int]bool{100: true, 101: true}
		}
		return map[int]bool{pid: true}
	}
	var procAt atomic.Int64
	r.swapper.sampleProcVRAMMB = func() (map[int]int64, time.Time, bool) {
		return map[int]int64{101: 4200, 555: 7777}, time.Unix(0, procAt.Load()), true
	}

	r.observeAndLearn() // empty host baseline
	a.setState(process.StateStarting)
	vramMB.Store(3000)
	r.observeAndLearn()
	a.markReady()
	vramMB.Store(5300)
	procAt.Store(time.Now().Add(time.Second).UnixNano())
	r.observeAndLearn()

	if est, ok := r.swapper.estimateMB("a"); !ok || est != 4200 {
		t.Fatalf("estimate=%d,%v want 4200 (per-process figure, not learned 9000)", est, ok)
	}
	hist := learnedHistory(r, "a")
	if len(hist) != 4 || hist[3] != 4200 {
		t.Fatalf("learned[a]=%v want 9000x3 then 4200 (per-process figure, not the 4300 delta)", hist)
	}

	// Another tick in the same load does not record again.
	r.observeAndLearn()
	if hist := learnedHistory(r, "a"); len(hist) != 4 {
		t.Fatalf("learned[a]=%v recorded twice for one load", hist)
	}

	// Stopped: the last per-process figure still estimates the next load.
	a.setState(process.StateStopped)
	r.observeAndLearn()
	if est, ok := r.swapper.estimateMB("a"); !ok || est != 4200 {
		t.Fatalf("estimate after stop=%d,%v want 4200", est, ok)
	}
}

// TestBudget_PerProcessIgnoresSampleBeforeReady checks that a per-process
// figure sampled before the model became ready (part-way through loading) is
// not used.
func TestBudget_PerProcessIgnoresSampleBeforeReady(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	processes := map[string]process.Process{"a": a}
	s := newTestBudgetSwapper(10000, nil, processes, 0)
	s.descendants = func(pid int) map[int]bool { return map[int]bool{pid: true} }
	s.sampleProcVRAMMB = func() (map[int]int64, time.Time, bool) {
		return map[int]int64{100: 1500}, time.Now().Add(-time.Hour), true
	}
	a.markReady()

	if got, _ := s.measureProcesses(map[string]time.Time{"a": time.Now()}); len(got) != 0 {
		t.Fatalf("measured=%v want none (sample predates ready)", got)
	}
	if _, ok := s.estimateMB("a"); ok {
		t.Fatal("estimate known from a stale per-process sample")
	}
}

// TestBudget_DeclaredSizeWinsOverMeasured checks the estimate order: declared
// sizeMB, then per-process measurement, then learned median.
func TestBudget_DeclaredSizeWinsOverMeasured(t *testing.T) {
	s := newTestBudgetSwapper(10000, nil, map[string]process.Process{}, 0)
	if _, ok := s.estimateMB("a"); ok {
		t.Fatal("estimate known with no data")
	}
	s.recordLearned([]string{"a"}, 1000)
	s.recordLearned([]string{"a"}, 1000)
	s.recordLearned([]string{"a"}, 1000)
	if est, _ := s.estimateMB("a"); est != 1000 {
		t.Fatalf("estimate=%d want learned 1000", est)
	}
	s.measured = map[string]int64{"a": 2000}
	if est, _ := s.estimateMB("a"); est != 2000 {
		t.Fatalf("estimate=%d want measured 2000", est)
	}
	s.declared = map[string]int64{"a": 3000}
	if est, _ := s.estimateMB("a"); est != 3000 {
		t.Fatalf("estimate=%d want declared 3000", est)
	}
}

// TestBudget_DeltaLearningStopsOnceAllMeasured checks that the delta-learning
// baseline is only kept while some configured model lacks a per-process
// figure, and that an unmatched model is still learned from deltas alongside
// a matched one.
func TestBudget_DeltaLearningStopsOnceAllMeasured(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	b := newFakeProcess("b")
	b.testPid = 200 // e.g. started via docker: not in nvidia-smi's PID list

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 100000}}}},
	}
	var vramMB atomic.Int64
	vramMB.Store(1000)
	r := newTestBudgetWithUsage(t, conf, nil, map[string]process.Process{"a": a, "b": b}, &vramMB)
	r.swapper.descendants = func(pid int) map[int]bool { return map[int]bool{pid: true} }
	var procAt atomic.Int64
	r.swapper.sampleProcVRAMMB = func() (map[int]int64, time.Time, bool) {
		return map[int]int64{100: 4000, 999: 1}, time.Unix(0, procAt.Load()), true
	}

	// a loads and is measured per process; b has never been seen, so the
	// baseline is still maintained.
	r.observeAndLearn()
	a.setState(process.StateStarting)
	r.observeAndLearn()
	a.markReady()
	vramMB.Store(5000)
	procAt.Store(time.Now().Add(time.Second).UnixNano())
	r.observeAndLearn()
	if r.baseline == nil {
		t.Fatal("baseline dropped while b still has no per-process figure")
	}

	// b loads; it has no PID match, so its footprint comes from the delta.
	b.setState(process.StateStarting)
	r.observeAndLearn()
	b.markReady()
	vramMB.Store(8000)
	r.observeAndLearn()
	if hist := learnedHistory(r, "b"); len(hist) != 1 || hist[0] != 3000 {
		t.Fatalf("learned[b]=%v want [3000] (delta 5000 -> 8000)", hist)
	}
	if hist := learnedHistory(r, "a"); len(hist) != 1 || hist[0] != 4000 {
		t.Fatalf("learned[a]=%v want [4000] (per-process only, no delta)", hist)
	}

	// Once every model has a per-process figure the baseline is dropped.
	r.swapper.measuredMu.Lock()
	r.swapper.measured["b"] = 3000
	r.swapper.measuredMu.Unlock()
	r.observeAndLearn()
	if r.baseline != nil {
		t.Fatal("baseline kept although every model has a per-process figure")
	}
}

// TestBudget_EnforceBudgetOncePausesWhenFreedLessThanPredicted checks that,
// with per-process figures for the evicted model, periodic eviction pauses
// when usage fell by much less than predicted, and carries on when it fell
// by about what was predicted.
func TestBudget_EnforceBudgetOncePausesWhenFreedLessThanPredicted(t *testing.T) {
	for _, tc := range []struct {
		name       string
		afterMB    int64
		wantBStops int32
	}{
		{"freed 500 of 8000: pause", 19500, 0},
		{"freed 4000 of 8000: continue", 16000, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := newFakeProcess("a")
			a.markReady()
			b := newFakeProcess("b")
			b.markReady()
			c := newFakeProcess("c")
			c.markReady()
			conf := config.Config{
				HealthCheckTimeout: 5,
				Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 15000}}}},
			}
			var used atomic.Int64
			used.Store(20000)
			r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 5, "c": 10},
				map[string]process.Process{"a": a, "b": b, "c": c}, &used)
			r.swapper.measured = map[string]int64{"a": 8000, "b": 8000, "c": 4000}

			// 20000 - 8000 (a) = 12000 fits: only a goes.
			r.enforceBudgetOnce()
			if a.stopCalls.Load() != 1 || b.stopCalls.Load() != 0 {
				t.Fatalf("first tick: a.stops=%d b.stops=%d want 1, 0", a.stopCalls.Load(), b.stopCalls.Load())
			}
			if r.lastEvict == nil || r.lastEvict.predictedMB != 8000 {
				t.Fatalf("lastEvict=%+v want predictedMB 8000", r.lastEvict)
			}

			used.Store(tc.afterMB)
			r.enforceBudgetOnce()
			if got := b.stopCalls.Load(); got != tc.wantBStops {
				t.Fatalf("second tick: b.stops=%d want %d", got, tc.wantBStops)
			}
		})
	}
}
