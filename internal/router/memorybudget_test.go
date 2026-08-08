package router

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/process"
)

// newTestMemoryBudgetSwapper builds a memoryBudgetSwapper with a stubbed VRAM
// sampler so tests can set memory usage deterministically without depending
// on real GPU monitoring tools.
func newTestMemoryBudgetSwapper(limitMB int64, priority map[string]int, processes map[string]process.Process, usedMB int64) *memoryBudgetSwapper {
	return &memoryBudgetSwapper{
		limitMB:   limitMB,
		priority:  priority,
		processes: processes,
		logger:    logmon.NewWriter(io.Discard),
		sampleVRAMUsedMB: func() (int64, bool) {
			return usedMB, true
		},
	}
}

func TestMemoryBudgetSwapper_AlreadyRunning(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	processes := map[string]process.Process{"a": a}
	s := newTestMemoryBudgetSwapper(1000, nil, processes, 1200)

	evict := s.EvictionFor("a", []string{"a"})
	if len(evict) != 0 {
		t.Errorf("evict=%v want none (already running)", evict)
	}
}

func TestMemoryBudgetSwapper_FitsWithoutEviction(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	processes := map[string]process.Process{"a": a}
	s := newTestMemoryBudgetSwapper(1000, nil, processes, 600)

	evict := s.EvictionFor("b", []string{"a"})
	if len(evict) != 0 {
		t.Errorf("evict=%v want none (usedMB under limit)", evict)
	}
}

func TestMemoryBudgetSwapper_NoSampleAvailable(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	processes := map[string]process.Process{"a": a}
	s := &memoryBudgetSwapper{
		limitMB:   1000,
		processes: processes,
		logger:    logmon.NewWriter(io.Discard),
		sampleVRAMUsedMB: func() (int64, bool) {
			return 0, false
		},
	}

	evict := s.EvictionFor("b", []string{"a"})
	if len(evict) != 0 {
		t.Errorf("evict=%v want none (no VRAM sample available)", evict)
	}
}

// Unified-memory hosts (NVIDIA Grace-Blackwell GB10, for example) report 0MB
// used to nvidia-smi because there is no separate framebuffer. The budget then
// applies to system memory used instead of reading the 0 as "nothing loaded".
func TestMemoryBudgetSwapper_FallsBackToSystemMemory(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	processes := map[string]process.Process{"a": a}
	s := newTestMemoryBudgetSwapper(1000, nil, processes, 0)
	s.sampleSysMemUsedMB = func() (int64, bool) { return 1200, true }

	evict := s.EvictionFor("b", []string{"a"})
	if len(evict) != 1 || evict[0] != "a" {
		t.Errorf("evict=%v want [a] (system memory over limit when VRAM reads 0)", evict)
	}
}

func TestMemoryBudgetSwapper_NoSampleFromEitherSource(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	processes := map[string]process.Process{"a": a}
	s := newTestMemoryBudgetSwapper(1000, nil, processes, 0)
	s.sampleSysMemUsedMB = func() (int64, bool) { return 0, false }

	evict := s.EvictionFor("b", []string{"a"})
	if len(evict) != 0 {
		t.Errorf("evict=%v want none (no memory sample available)", evict)
	}
}

func TestMemoryBudgetSwapper_EvictsLowestPriorityFirst(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()
	c := newFakeProcess("c")
	c.testPid = 102
	c.markReady()
	processes := map[string]process.Process{"a": a, "b": b, "c": c}
	priority := map[string]int{"a": 10, "b": 5, "c": 1}
	// 1500MB used across 3 running models (500MB share each) over a 1000MB
	// cap: evicting the lowest-priority model (c) alone is enough to fit.
	s := newTestMemoryBudgetSwapper(1000, priority, processes, 1500)

	evict := s.EvictionFor("d", []string{"a", "b", "c"})
	if len(evict) != 1 || evict[0] != "c" {
		t.Fatalf("evict=%v want [c]", evict)
	}
}

func TestMemoryBudgetSwapper_EvictsThroughHigherPriorityWhenNeeded(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()
	processes := map[string]process.Process{"a": a, "b": b}
	priority := map[string]int{"a": 10, "b": 1}
	// 2500MB used across 2 running models (1250MB share each) over a 1000MB
	// cap: evicting only the lowest-priority model (b) leaves 1250MB, still
	// over the cap, so eviction must proceed through a too.
	s := newTestMemoryBudgetSwapper(1000, priority, processes, 2500)

	evict := s.EvictionFor("c", []string{"a", "b"})
	if len(evict) != 2 {
		t.Fatalf("evict=%v want both models evicted to fit", evict)
	}
}

func TestMemoryBudgetSwapper_TieBreakLRU(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	a.lastUse = time.Now().Add(-time.Hour)
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()
	b.lastUse = time.Now()
	processes := map[string]process.Process{"a": a, "b": b}
	priority := map[string]int{"a": 5, "b": 5} // tied priority
	// 1200MB used across 2 running models (600MB share each) over a 1000MB
	// cap: evicting the older model (a) alone is enough to fit.
	s := newTestMemoryBudgetSwapper(1000, priority, processes, 1200)

	// Equal priority: the least-recently-used (a) is evicted first.
	evict := s.EvictionFor("c", []string{"a", "b"})
	if len(evict) != 1 || evict[0] != "a" {
		t.Fatalf("evict=%v want [a] (older lastUse)", evict)
	}
}

// newTestMemoryBudget builds a MemoryBudget router from supplied processes,
// bypassing NewMemoryBudget's call to process.New, mirroring newTestMatrix.
func newTestMemoryBudget(t *testing.T, conf config.Config, priority map[string]int, processes map[string]process.Process, usedMB int64) *MemoryBudget {
	t.Helper()
	swapper := newTestMemoryBudgetSwapper(conf.MemoryBudget.Limit.MB(), priority, processes, usedMB)
	base, err := newBaseRouter("memoryBudget", conf, processes, swapper.logger, swapper)
	if err != nil {
		t.Fatalf("newBaseRouter: %v", err)
	}
	base.testProcessed = make(chan struct{}, 64)
	r := &MemoryBudget{baseRouter: base, swapper: swapper}
	go base.run()
	t.Cleanup(func() {
		if !r.shuttingDown.Load() {
			_ = r.Shutdown(time.Second)
		}
	})
	return r
}

// newTestMemoryBudgetWithUsage builds a MemoryBudget whose memory sample can be
// changed between calls, for exercising enforceBudgetOnce across ticks.
func newTestMemoryBudgetWithUsage(t *testing.T, conf config.Config, priority map[string]int, processes map[string]process.Process, usedMB *atomic.Int64) *MemoryBudget {
	t.Helper()
	r := newTestMemoryBudget(t, conf, priority, processes, 0)
	r.swapper.sampleVRAMUsedMB = func() (int64, bool) { return usedMB.Load(), true }
	return r
}

func TestMemoryBudget_EnforceBudgetOnceUnderBudget(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()

	conf := config.Config{
		HealthCheckTimeout: 5,
		MemoryBudget:       &config.MemoryBudgetConfig{Limit: 1000},
	}
	var usedMB atomic.Int64
	usedMB.Store(600)
	r := newTestMemoryBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 10}, map[string]process.Process{"a": a, "b": b}, &usedMB)

	r.enforceBudgetOnce()

	if got := a.stopCalls.Load(); got != 0 {
		t.Errorf("a.stopCalls=%d want 0 (under budget)", got)
	}
	if got := b.stopCalls.Load(); got != 0 {
		t.Errorf("b.stopCalls=%d want 0 (under budget)", got)
	}
}

func TestMemoryBudget_EnforceBudgetOnceEvictsLowestPriority(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()
	c := newFakeProcess("c")
	c.testPid = 102
	c.markReady()

	conf := config.Config{
		HealthCheckTimeout: 5,
		MemoryBudget:       &config.MemoryBudgetConfig{Limit: 1000},
	}
	// 1200MB used across 3 running models (400MB share each) over a 1000MB
	// cap: evicting the lowest-priority model (a) alone is enough to fit.
	var usedMB atomic.Int64
	usedMB.Store(1200)
	processes := map[string]process.Process{"a": a, "b": b, "c": c}
	r := newTestMemoryBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 10, "c": 10}, processes, &usedMB)

	r.enforceBudgetOnce()

	if got := a.stopCalls.Load(); got != 1 {
		t.Errorf("a.stopCalls=%d want 1 (lowest priority, over budget with no swap)", got)
	}
	if got := b.stopCalls.Load(); got != 0 {
		t.Errorf("b.stopCalls=%d want 0", got)
	}
	if got := c.stopCalls.Load(); got != 0 {
		t.Errorf("c.stopCalls=%d want 0", got)
	}
}

func TestMemoryBudget_EnforceBudgetOnceSkipsStartingModels(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.setState(process.StateStarting)

	conf := config.Config{
		HealthCheckTimeout: 5,
		MemoryBudget:       &config.MemoryBudgetConfig{Limit: 1000},
	}
	// b has the lower priority, so it would be evicted first if it were
	// eligible — but it is still starting, leaving only a as a candidate.
	var usedMB atomic.Int64
	usedMB.Store(1200)
	processes := map[string]process.Process{"a": a, "b": b}
	r := newTestMemoryBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 0}, processes, &usedMB)

	r.enforceBudgetOnce()

	if got := a.stopCalls.Load(); got != 1 {
		t.Errorf("a.stopCalls=%d want 1 (only ready candidate)", got)
	}
	if got := b.stopCalls.Load(); got != 0 {
		t.Errorf("b.stopCalls=%d want 0 (still starting, not evictable)", got)
	}
}

func TestMemoryBudget_EnforceBudgetOnceKeepsLastModel(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()

	conf := config.Config{
		HealthCheckTimeout: 5,
		MemoryBudget:       &config.MemoryBudgetConfig{Limit: 1000},
	}
	// One model whose own footprint blows the budget. Evicting it would empty
	// the host, and the next request would just reload and re-evict it.
	var usedMB atomic.Int64
	usedMB.Store(5000)
	r := newTestMemoryBudgetWithUsage(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a}, &usedMB)

	r.enforceBudgetOnce()
	r.enforceBudgetOnce()

	if got := a.stopCalls.Load(); got != 0 {
		t.Errorf("a.stopCalls=%d want 0 (last running model is never evicted)", got)
	}
}

func TestMemoryBudget_EnforceBudgetOnceStopsWhenEvictionFreesNothing(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()
	c := newFakeProcess("c")
	c.testPid = 102
	c.markReady()

	conf := config.Config{
		HealthCheckTimeout: 5,
		MemoryBudget:       &config.MemoryBudgetConfig{Limit: 1000},
	}
	// Usage never moves, as on a unified-memory host where the sample counts
	// every process rather than only models.
	var usedMB atomic.Int64
	usedMB.Store(1200)
	processes := map[string]process.Process{"a": a, "b": b, "c": c}
	r := newTestMemoryBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 2, "c": 3}, processes, &usedMB)

	r.enforceBudgetOnce()
	if got := a.stopCalls.Load(); got != 1 {
		t.Fatalf("a.stopCalls=%d want 1 on the first tick", got)
	}

	// Second tick: still over budget, but evicting a freed nothing, so no
	// further model may be evicted.
	r.enforceBudgetOnce()
	if got := b.stopCalls.Load(); got != 0 {
		t.Errorf("b.stopCalls=%d want 0 (previous eviction freed nothing)", got)
	}

	// Usage finally drops while staying over budget: eviction is trusted
	// again.
	usedMB.Store(1100)
	r.enforceBudgetOnce()
	if got := b.stopCalls.Load(); got != 1 {
		t.Errorf("b.stopCalls=%d want 1 (usage dropped, eviction resumes)", got)
	}
	if got := c.stopCalls.Load(); got != 0 {
		t.Errorf("c.stopCalls=%d want 0", got)
	}
}

func TestMemoryBudget_SwapEvictsLowerPriority(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	go a.Run(0)

	b := newFakeProcess("b")
	b.testPid = 101
	b.autoReady = true

	conf := config.Config{
		HealthCheckTimeout: 5,
		MemoryBudget:       &config.MemoryBudgetConfig{Limit: 1000},
	}
	priority := map[string]int{"a": 1, "b": 10}

	r := newTestMemoryBudget(t, conf, priority, map[string]process.Process{"a": a, "b": b}, 1800)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, newRequest("b"))

	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
	}
	if got := a.stopCalls.Load(); got != 1 {
		t.Errorf("a.stopCalls=%d want 1 (lower priority, evicted to fit budget)", got)
	}
	if got := b.runCalls.Load(); got != 1 {
		t.Errorf("b.runCalls=%d want 1", got)
	}
}

// TestMemoryBudgetSwapper_ReservesForInFlightLoads covers the concurrent-load
// blind spot: EvictionFor's usedMB sample never includes a model that hasn't
// finished loading yet, so a burst of concurrent loads can each individually
// look like they fit. A learned footprint for a still-loading model closes
// that gap for repeat loads of the same model.
func TestMemoryBudgetSwapper_ReservesForInFlightLoads(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.setState(process.StateStarting) // committed (in running) but not yet resident

	processes := map[string]process.Process{"a": a, "b": b}
	priority := map[string]int{"a": 1, "b": 10}
	// usedMB=600 reflects only "a" — "b" is mid-load and hasn't shown up in
	// the sample yet. Without a reservation this looks like it fits under
	// the 800 limit.
	s := newTestMemoryBudgetSwapper(800, priority, processes, 600)
	// b previously observed costing ~500MB, enough times to be trusted.
	s.recordLearned([]string{"b"}, 500)
	s.recordLearned([]string{"b"}, 500)
	s.recordLearned([]string{"b"}, 500)

	evict := s.EvictionFor("c", []string{"a", "b"})
	if len(evict) != 1 || evict[0] != "a" {
		t.Fatalf("evict=%v want [a] (600 + b's reserved 500 = 1100, over the 800 limit)", evict)
	}
}

// TestMemoryBudgetSwapper_NoReservationBelowMinSamples checks that a handful
// of observations aren't trusted yet: reservedForInFlightMB requires
// minLearnedSamples before using the median, so noise in the first one or two
// loads can't drive a reservation on its own.
func TestMemoryBudgetSwapper_NoReservationBelowMinSamples(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.setState(process.StateStarting)

	processes := map[string]process.Process{"a": a, "b": b}
	s := newTestMemoryBudgetSwapper(800, map[string]int{"a": 1, "b": 10}, processes, 600)
	// Only 2 observations recorded; minLearnedSamples is 3.
	s.recordLearned([]string{"b"}, 500)
	s.recordLearned([]string{"b"}, 500)

	evict := s.EvictionFor("c", []string{"a", "b"})
	if len(evict) != 0 {
		t.Fatalf("evict=%v want none (only 2 observations, below minLearnedSamples=3, so b reserves nothing yet)", evict)
	}
}

// TestMemoryBudgetSwapper_MedianRejectsOutlier checks the actual point of
// keeping a history instead of overwriting: a single noisy observation (the
// unified-memory fallback attributing an unrelated host allocation to a
// model) must not dominate the reservation the way it would if only the
// latest value were trusted.
func TestMemoryBudgetSwapper_MedianRejectsOutlier(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.setState(process.StateStarting)

	processes := map[string]process.Process{"a": a, "b": b}
	s := newTestMemoryBudgetSwapper(1000, map[string]int{"a": 1, "b": 10}, processes, 600)
	// b's real footprint is consistently ~200MB, but one load coincided with
	// an unrelated 5000MB spike elsewhere on the host and got misattributed.
	s.recordLearned([]string{"b"}, 200)
	s.recordLearned([]string{"b"}, 5000)
	s.recordLearned([]string{"b"}, 210)
	s.recordLearned([]string{"b"}, 195)

	// Median of [200, 5000, 210, 195] sorted -> [195, 200, 210, 5000] -> (200+210)/2 = 205.
	// 600 + 205 = 805, under the 1000 limit: the outlier must not push this
	// over, which trusting only the latest (5000) value would have done.
	evict := s.EvictionFor("c", []string{"a", "b"})
	if len(evict) != 0 {
		t.Fatalf("evict=%v want none (median of b's history is ~205MB; the 5000MB outlier must be outvoted, not trusted)", evict)
	}
}

// TestMemoryBudgetSwapper_NoReservationWithoutPriorObservation documents the
// remaining blind spot: a model that has never finished loading before has no
// learned entry, so it contributes no reservation and the very first
// concurrent load of it is still invisible to EvictionFor.
func TestMemoryBudgetSwapper_NoReservationWithoutPriorObservation(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.setState(process.StateStarting)

	processes := map[string]process.Process{"a": a, "b": b}
	s := newTestMemoryBudgetSwapper(800, map[string]int{"a": 1, "b": 10}, processes, 600)
	// No recordLearned call for b: this is its first-ever observed load.

	evict := s.EvictionFor("c", []string{"a", "b"})
	if len(evict) != 0 {
		t.Fatalf("evict=%v want none (b has no learned footprint yet, so it reserves nothing)", evict)
	}
}

// TestMemoryBudgetSwapper_ReadyModelsAreNotReserved checks that only
// not-yet-ready models draw a reservation — a model already counted in usedMB
// must not also have its learned footprint added on top.
func TestMemoryBudgetSwapper_ReadyModelsAreNotReserved(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()

	processes := map[string]process.Process{"a": a}
	s := newTestMemoryBudgetSwapper(800, map[string]int{"a": 1}, processes, 600)
	// stale learned history from past loads
	s.recordLearned([]string{"a"}, 500)
	s.recordLearned([]string{"a"}, 500)
	s.recordLearned([]string{"a"}, 500)

	evict := s.EvictionFor("c", []string{"a"})
	if len(evict) != 0 {
		t.Fatalf("evict=%v want none (a is already ready and counted in usedMB=600; its learned footprint must not be double-counted)", evict)
	}
}

// TestMemoryBudgetSwapper_StoppingModelsAreNotReserved checks that a model on
// its way out is not reserved for. runningSet includes StateStopping models,
// whose memory is still in the usedMB sample and about to be freed — adding a
// reservation on top would double-count them and cascade into evicting more
// than the budget actually calls for.
func TestMemoryBudgetSwapper_StoppingModelsAreNotReserved(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.setState(process.StateStopping) // already being evicted by another swap

	processes := map[string]process.Process{"a": a, "b": b}
	s := newTestMemoryBudgetSwapper(1000, map[string]int{"a": 1, "b": 10}, processes, 900)
	s.recordLearned([]string{"b"}, 800)
	s.recordLearned([]string{"b"}, 800)
	s.recordLearned([]string{"b"}, 800)

	evict := s.EvictionFor("c", []string{"a", "b"})
	if len(evict) != 0 {
		t.Fatalf("evict=%v want none (b is stopping: its 800MB is already in usedMB=900 and about to be freed, not a load to reserve for)", evict)
	}
}

// TestMemoryBudgetSwapper_ShareExcludesReservation checks that the assumed
// per-eviction yield comes from the sampled usage only. Reserved memory
// belongs to models that have not loaded yet, so evicting a ready model
// cannot free any of it.
func TestMemoryBudgetSwapper_ShareExcludesReservation(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()
	c := newFakeProcess("c")
	c.testPid = 102
	c.setState(process.StateStarting)

	processes := map[string]process.Process{"a": a, "b": b, "c": c}
	priority := map[string]int{"a": 1, "b": 2, "c": 10}
	// sampled=900 over 3 running models is a 300MB share each; c reserves a
	// further 600MB, for 1500MB against a 1000MB limit. Evicting a alone
	// frees ~300 (1200, still over), so b must go too. Folding the
	// reservation into the share would make each eviction look like it frees
	// 500MB and stop after a.
	s := newTestMemoryBudgetSwapper(1000, priority, processes, 900)
	s.recordLearned([]string{"c"}, 600)
	s.recordLearned([]string{"c"}, 600)
	s.recordLearned([]string{"c"}, 600)

	evict := s.EvictionFor("d", []string{"a", "b", "c"})
	if len(evict) != 2 || evict[0] != "a" || evict[1] != "b" {
		t.Fatalf("evict=%v want [a b] (300MB share from the 900MB sample, not 500MB from sample+reservation)", evict)
	}
}

// TestMemoryBudget_ObserveAndLearnRecordsFromSysMemSource checks that the
// unified-memory fallback source is not blocked from learning outright — the
// sample it produces counts every process on the host, not only models, so a
// single observation from it can be noise, but observeAndLearn still records
// it into the model's history. Robustness against that noise comes from
// reservedForInFlightMB's median-over-history logic (see
// TestMemoryBudgetSwapper_MedianRejectsOutlier and
// TestMemoryBudgetSwapper_NoReservationBelowMinSamples), not from refusing to
// learn from this source at all — that would throw away real signal along
// with the noise on any host where VRAM reporting never works (which on a
// unified-memory host is every host).
func TestMemoryBudget_ObserveAndLearnRecordsFromSysMemSource(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.setState(process.StateStarting)

	conf := config.Config{
		HealthCheckTimeout: 5,
		MemoryBudget:       &config.MemoryBudgetConfig{Limit: 1000},
	}
	var usedMB atomic.Int64
	usedMB.Store(100)
	r := newTestMemoryBudgetWithUsage(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a}, &usedMB)
	// VRAM reads 0, so usedMemoryMB falls back to system memory.
	r.swapper.sampleVRAMUsedMB = func() (int64, bool) { return 0, true }
	r.swapper.sampleSysMemUsedMB = func() (int64, bool) { return usedMB.Load(), true }

	r.observeAndLearn()

	a.markReady()
	usedMB.Store(400) // could be the model, could be anything else on the host
	r.observeAndLearn()

	r.swapper.learnedMu.RLock()
	hist := r.swapper.learned["a"]
	r.swapper.learnedMu.RUnlock()
	if len(hist) != 1 || hist[0] != 300 {
		t.Fatalf("learned[a]=%v want [300] (400-100 delta recorded even from a system-memory sample)", hist)
	}
}

func TestMemoryBudget_ObserveAndLearnAttributesDelta(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.setState(process.StateStarting)

	conf := config.Config{
		HealthCheckTimeout: 5,
		MemoryBudget:       &config.MemoryBudgetConfig{Limit: 1000},
	}
	var usedMB atomic.Int64
	usedMB.Store(100)
	r := newTestMemoryBudgetWithUsage(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a}, &usedMB)

	r.observeAndLearn() // first tick: establishes the baseline, a not ready yet

	a.markReady()
	usedMB.Store(400)
	r.observeAndLearn() // second tick: a transitioned to ready, delta=300

	r.swapper.learnedMu.RLock()
	hist := r.swapper.learned["a"]
	r.swapper.learnedMu.RUnlock()
	if len(hist) != 1 || hist[0] != 300 {
		t.Fatalf("learned[a]=%v want [300] (400-100 delta attributed to a)", hist)
	}
}

func TestMemoryBudget_ObserveAndLearnSplitsDeltaAcrossConcurrentLoads(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.setState(process.StateStarting)
	b := newFakeProcess("b")
	b.testPid = 101
	b.setState(process.StateStarting)

	conf := config.Config{
		HealthCheckTimeout: 5,
		MemoryBudget:       &config.MemoryBudgetConfig{Limit: 1000},
	}
	var usedMB atomic.Int64
	usedMB.Store(100)
	r := newTestMemoryBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 1}, map[string]process.Process{"a": a, "b": b}, &usedMB)

	r.observeAndLearn()

	a.markReady()
	b.markReady()
	usedMB.Store(700) // both became ready between ticks; 600 delta split evenly

	r.observeAndLearn()

	r.swapper.learnedMu.RLock()
	defer r.swapper.learnedMu.RUnlock()
	histA, histB := r.swapper.learned["a"], r.swapper.learned["b"]
	if len(histA) != 1 || histA[0] != 300 || len(histB) != 1 || histB[0] != 300 {
		t.Fatalf("learned a=%v b=%v want a=[300] b=[300] (600 delta split across 2 concurrent loads)", histA, histB)
	}
}

func TestMemoryBudget_ObserveAndLearnSkipsNonPositiveDelta(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.setState(process.StateStarting)

	conf := config.Config{
		HealthCheckTimeout: 5,
		MemoryBudget:       &config.MemoryBudgetConfig{Limit: 1000},
	}
	var usedMB atomic.Int64
	usedMB.Store(500)
	r := newTestMemoryBudgetWithUsage(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a}, &usedMB)

	r.observeAndLearn()

	a.markReady()
	usedMB.Store(300) // usage fell between ticks (e.g. something else was evicted)

	r.observeAndLearn()

	r.swapper.learnedMu.RLock()
	_, ok := r.swapper.learned["a"]
	r.swapper.learnedMu.RUnlock()
	if ok {
		t.Fatalf("learned[a] should not be recorded from a non-positive delta")
	}
}

// TestMemoryBudget_EnforceBudgetOnceResumesWhenNewLoadExplainsTheRise covers
// the false-pause bug: usage rising since the last eviction does not mean the
// eviction failed if a concurrent load finished in between and explains the
// rise on its own. Pausing in that case disables protection for as long as
// the burst continues — exactly when it's needed most.
func TestMemoryBudget_EnforceBudgetOnceResumesWhenNewLoadExplainsTheRise(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()
	c := newFakeProcess("c")
	c.testPid = 102
	c.setState(process.StateStarting) // not ready at the first tick

	conf := config.Config{
		HealthCheckTimeout: 5,
		MemoryBudget:       &config.MemoryBudgetConfig{Limit: 1000},
	}
	var usedMB atomic.Int64
	usedMB.Store(1200)
	processes := map[string]process.Process{"a": a, "b": b, "c": c}
	r := newTestMemoryBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 2, "c": 3}, processes, &usedMB)

	r.enforceBudgetOnce()
	if got := a.stopCalls.Load(); got != 1 {
		t.Fatalf("a.stopCalls=%d want 1 on the first tick", got)
	}

	// Second tick: usage rose instead of dropping, but "c" finished loading
	// in between (a legitimate new member of the ready set) — that alone
	// explains the rise, so eviction must not be treated as having failed.
	c.markReady()
	usedMB.Store(1400)
	r.enforceBudgetOnce()
	if got := b.stopCalls.Load(); got != 1 {
		t.Errorf("b.stopCalls=%d want 1 (c finishing loading explains the rise, so eviction resumes rather than pausing)", got)
	}
}

// TestMemoryBudget_NextCheckIntervalTightensWhileLoading checks that the
// periodic check switches to the tighter interval whenever any model is
// mid-load, and relaxes back once nothing is.
func TestMemoryBudget_NextCheckIntervalTightensWhileLoading(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()

	conf := config.Config{
		HealthCheckTimeout: 5,
		MemoryBudget:       &config.MemoryBudgetConfig{Limit: 1000},
	}
	r := newTestMemoryBudget(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a}, 100)

	if got := r.nextCheckInterval(); got != memoryBudgetCheckInterval {
		t.Errorf("nextCheckInterval=%v want %v (nothing loading)", got, memoryBudgetCheckInterval)
	}

	b := newFakeProcess("b")
	b.testPid = 101
	b.setState(process.StateStarting)
	r.processes["b"] = b

	if got := r.nextCheckInterval(); got != memoryBudgetActiveLoadCheckInterval {
		t.Errorf("nextCheckInterval=%v want %v (b is loading)", got, memoryBudgetActiveLoadCheckInterval)
	}

	b.markReady()
	if got := r.nextCheckInterval(); got != memoryBudgetCheckInterval {
		t.Errorf("nextCheckInterval=%v want %v (b finished loading)", got, memoryBudgetCheckInterval)
	}
}
