package router

import (
	"bytes"
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

// newTestBudgetSwapper builds a budgetSwapper with a stubbed VRAM
// sampler so tests can set memory usage deterministically without depending
// on real GPU monitoring tools.
func newTestBudgetSwapper(limitMB int64, priority map[string]int, processes map[string]process.Process, usedMB int64) *budgetSwapper {
	return &budgetSwapper{
		limitMB:   limitMB,
		priority:  priority,
		processes: processes,
		logger:    logmon.NewWriter(io.Discard),
		sampleVRAMUsedMB: func() (int64, time.Time, bool) {
			return usedMB, time.Now(), true
		},
	}
}

func TestBudgetSwapper_AlreadyRunning(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	processes := map[string]process.Process{"a": a}
	s := newTestBudgetSwapper(1000, nil, processes, 1200)

	evict := s.EvictionFor("a", []string{"a"})
	if len(evict) != 0 {
		t.Errorf("evict=%v want none (already running)", evict)
	}
}

func TestBudgetSwapper_FitsWithoutEviction(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	processes := map[string]process.Process{"a": a}
	s := newTestBudgetSwapper(1000, nil, processes, 600)

	evict := s.EvictionFor("b", []string{"a"})
	if len(evict) != 0 {
		t.Errorf("evict=%v want none (usedMB under limit)", evict)
	}
}

func TestBudgetSwapper_NoSampleAvailable(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	processes := map[string]process.Process{"a": a}
	s := &budgetSwapper{
		limitMB:   1000,
		processes: processes,
		logger:    logmon.NewWriter(io.Discard),
		sampleVRAMUsedMB: func() (int64, time.Time, bool) {
			return 0, time.Time{}, false
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
func TestBudgetSwapper_FallsBackToSystemMemory(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	processes := map[string]process.Process{"a": a}
	s := newTestBudgetSwapper(1000, nil, processes, 0)
	s.sampleSysMemUsedMB = func() (int64, time.Time, bool) { return 1200, time.Now(), true }

	evict := s.EvictionFor("b", []string{"a"})
	if len(evict) != 1 || evict[0] != "a" {
		t.Errorf("evict=%v want [a] (system memory over limit when VRAM reads 0)", evict)
	}
}

func TestBudgetSwapper_NoSampleFromEitherSource(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	processes := map[string]process.Process{"a": a}
	s := newTestBudgetSwapper(1000, nil, processes, 0)
	s.sampleSysMemUsedMB = func() (int64, time.Time, bool) { return 0, time.Time{}, false }

	evict := s.EvictionFor("b", []string{"a"})
	if len(evict) != 0 {
		t.Errorf("evict=%v want none (no memory sample available)", evict)
	}
}

func TestBudgetSwapper_EvictsLowestPriorityFirst(t *testing.T) {
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
	s := newTestBudgetSwapper(1000, priority, processes, 1500)

	evict := s.EvictionFor("d", []string{"a", "b", "c"})
	if len(evict) != 1 || evict[0] != "c" {
		t.Fatalf("evict=%v want [c]", evict)
	}
}

func TestBudgetSwapper_EvictsThroughHigherPriorityWhenNeeded(t *testing.T) {
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
	s := newTestBudgetSwapper(1000, priority, processes, 2500)

	evict := s.EvictionFor("c", []string{"a", "b"})
	if len(evict) != 2 {
		t.Fatalf("evict=%v want both models evicted to fit", evict)
	}
}

func TestBudgetSwapper_TieBreakLRU(t *testing.T) {
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
	s := newTestBudgetSwapper(1000, priority, processes, 1200)

	// Equal priority: the least-recently-used (a) is evicted first.
	evict := s.EvictionFor("c", []string{"a", "b"})
	if len(evict) != 1 || evict[0] != "a" {
		t.Fatalf("evict=%v want [a] (older lastUse)", evict)
	}
}

// newTestBudget builds a Budget router from supplied processes,
// bypassing NewBudget's call to process.New, mirroring newTestMatrix.
func newTestBudget(t *testing.T, conf config.Config, priority map[string]int, processes map[string]process.Process, usedMB int64) *Budget {
	t.Helper()
	swapper := newTestBudgetSwapper(conf.Routing.Router.Settings.Budget.LimitMB, priority, processes, usedMB)
	base, err := newBaseRouter("budget", conf, processes, swapper.logger, swapper)
	if err != nil {
		t.Fatalf("newBaseRouter: %v", err)
	}
	base.testProcessed = make(chan struct{}, 64)
	r := &Budget{baseRouter: base, swapper: swapper}
	go base.run()
	t.Cleanup(func() {
		if !r.shuttingDown.Load() {
			_ = r.Shutdown(time.Second)
		}
	})
	return r
}

// newTestBudgetWithUsage builds a Budget whose memory sample can be
// changed between calls, for exercising enforceBudgetOnce across ticks.
func newTestBudgetWithUsage(t *testing.T, conf config.Config, priority map[string]int, processes map[string]process.Process, usedMB *atomic.Int64) *Budget {
	t.Helper()
	r := newTestBudget(t, conf, priority, processes, 0)
	r.swapper.sampleVRAMUsedMB = func() (int64, time.Time, bool) { return usedMB.Load(), time.Now(), true }
	return r
}

func TestBudget_EnforceBudgetOnceUnderBudget(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	var usedMB atomic.Int64
	usedMB.Store(600)
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 10}, map[string]process.Process{"a": a, "b": b}, &usedMB)

	r.enforceBudgetOnce()

	if got := a.stopCalls.Load(); got != 0 {
		t.Errorf("a.stopCalls=%d want 0 (under budget)", got)
	}
	if got := b.stopCalls.Load(); got != 0 {
		t.Errorf("b.stopCalls=%d want 0 (under budget)", got)
	}
}

func TestBudget_EnforceBudgetOnceEvictsLowestPriority(t *testing.T) {
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
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	// 1200MB used across 3 running models (400MB share each) over a 1000MB
	// cap: evicting the lowest-priority model (a) alone is enough to fit.
	var usedMB atomic.Int64
	usedMB.Store(1200)
	processes := map[string]process.Process{"a": a, "b": b, "c": c}
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 10, "c": 10}, processes, &usedMB)

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

func TestBudget_EnforceBudgetOnceSkipsStartingModels(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.setState(process.StateStarting)

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	// b has the lower priority, so it would be evicted first if it were
	// eligible — but it is still starting, leaving only a as a candidate.
	var usedMB atomic.Int64
	usedMB.Store(1200)
	processes := map[string]process.Process{"a": a, "b": b}
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 0}, processes, &usedMB)

	r.enforceBudgetOnce()

	if got := a.stopCalls.Load(); got != 1 {
		t.Errorf("a.stopCalls=%d want 1 (only ready candidate)", got)
	}
	if got := b.stopCalls.Load(); got != 0 {
		t.Errorf("b.stopCalls=%d want 0 (still starting, not evictable)", got)
	}
}

func TestBudget_EnforceBudgetOnceKeepsLastModel(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	// One model whose own footprint blows the budget. Evicting it would empty
	// the host, and the next request would just reload and re-evict it.
	var usedMB atomic.Int64
	usedMB.Store(5000)
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a}, &usedMB)

	r.enforceBudgetOnce()
	r.enforceBudgetOnce()

	if got := a.stopCalls.Load(); got != 0 {
		t.Errorf("a.stopCalls=%d want 0 (last running model is never evicted)", got)
	}
}

func TestBudget_EnforceBudgetOnceStopsWhenEvictionFreesNothing(t *testing.T) {
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
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	// Usage never moves, as on a unified-memory host where the sample counts
	// every process rather than only models.
	var usedMB atomic.Int64
	usedMB.Store(1200)
	processes := map[string]process.Process{"a": a, "b": b, "c": c}
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 2, "c": 3}, processes, &usedMB)

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

func TestBudget_SwapEvictsLowerPriority(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	go a.Run(0)

	b := newFakeProcess("b")
	b.testPid = 101
	b.autoReady = true

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	priority := map[string]int{"a": 1, "b": 10}

	r := newTestBudget(t, conf, priority, map[string]process.Process{"a": a, "b": b}, 1800)

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

// TestBudgetSwapper_ReservesForInFlightLoads covers the concurrent-load
// blind spot: EvictionFor's usedMB sample never includes a model that hasn't
// finished loading yet, so a burst of concurrent loads can each individually
// look like they fit. A learned footprint for a still-loading model closes
// that gap for repeat loads of the same model.
func TestBudgetSwapper_ReservesForInFlightLoads(t *testing.T) {
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
	s := newTestBudgetSwapper(800, priority, processes, 600)
	// b previously observed costing ~500MB, enough times to be trusted.
	s.recordLearned([]string{"b"}, 500)
	s.recordLearned([]string{"b"}, 500)
	s.recordLearned([]string{"b"}, 500)

	evict := s.EvictionFor("c", []string{"a", "b"})
	if len(evict) != 1 || evict[0] != "a" {
		t.Fatalf("evict=%v want [a] (600 + b's reserved 500 = 1100, over the 800 limit)", evict)
	}
}

// TestBudgetSwapper_NoReservationBelowMinSamples checks that a handful
// of observations aren't trusted yet: reservedForInFlightMB requires
// minLearnedSamples before using the median, so noise in the first one or two
// loads can't drive a reservation on its own.
func TestBudgetSwapper_NoReservationBelowMinSamples(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.setState(process.StateStarting)

	processes := map[string]process.Process{"a": a, "b": b}
	s := newTestBudgetSwapper(800, map[string]int{"a": 1, "b": 10}, processes, 600)
	// Only 2 observations recorded; minLearnedSamples is 3.
	s.recordLearned([]string{"b"}, 500)
	s.recordLearned([]string{"b"}, 500)

	evict := s.EvictionFor("c", []string{"a", "b"})
	if len(evict) != 0 {
		t.Fatalf("evict=%v want none (only 2 observations, below minLearnedSamples=3, so b reserves nothing yet)", evict)
	}
}

// TestBudgetSwapper_MedianRejectsOutlier checks the actual point of
// keeping a history instead of overwriting: a single noisy observation (the
// unified-memory fallback attributing an unrelated host allocation to a
// model) must not dominate the reservation the way it would if only the
// latest value were trusted.
func TestBudgetSwapper_MedianRejectsOutlier(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.setState(process.StateStarting)

	processes := map[string]process.Process{"a": a, "b": b}
	s := newTestBudgetSwapper(1000, map[string]int{"a": 1, "b": 10}, processes, 600)
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

// TestBudgetSwapper_NoReservationWithoutPriorObservation documents the
// remaining blind spot: a model that has never finished loading before has no
// learned entry, so it contributes no reservation and the very first
// concurrent load of it is still invisible to EvictionFor.
func TestBudgetSwapper_NoReservationWithoutPriorObservation(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.setState(process.StateStarting)

	processes := map[string]process.Process{"a": a, "b": b}
	s := newTestBudgetSwapper(800, map[string]int{"a": 1, "b": 10}, processes, 600)
	// No recordLearned call for b: this is its first-ever observed load.

	evict := s.EvictionFor("c", []string{"a", "b"})
	if len(evict) != 0 {
		t.Fatalf("evict=%v want none (b has no learned footprint yet, so it reserves nothing)", evict)
	}
}

// TestBudgetSwapper_ReadyModelsAreNotReserved checks that only
// not-yet-ready models draw a reservation — a model already counted in usedMB
// must not also have its learned footprint added on top.
func TestBudgetSwapper_ReadyModelsAreNotReserved(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()

	processes := map[string]process.Process{"a": a}
	s := newTestBudgetSwapper(800, map[string]int{"a": 1}, processes, 600)
	// stale learned history from past loads
	s.recordLearned([]string{"a"}, 500)
	s.recordLearned([]string{"a"}, 500)
	s.recordLearned([]string{"a"}, 500)

	evict := s.EvictionFor("c", []string{"a"})
	if len(evict) != 0 {
		t.Fatalf("evict=%v want none (a is already ready and counted in usedMB=600; its learned footprint must not be double-counted)", evict)
	}
}

// TestBudgetSwapper_StoppingModelsAreNotReserved checks that a model on
// its way out is not reserved for. runningSet includes StateStopping models,
// whose memory is still in the usedMB sample and about to be freed — adding a
// reservation on top would double-count them and cascade into evicting more
// than the budget actually calls for.
func TestBudgetSwapper_StoppingModelsAreNotReserved(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.setState(process.StateStopping) // already being evicted by another swap

	processes := map[string]process.Process{"a": a, "b": b}
	s := newTestBudgetSwapper(1000, map[string]int{"a": 1, "b": 10}, processes, 900)
	s.recordLearned([]string{"b"}, 800)
	s.recordLearned([]string{"b"}, 800)
	s.recordLearned([]string{"b"}, 800)

	evict := s.EvictionFor("c", []string{"a", "b"})
	if len(evict) != 0 {
		t.Fatalf("evict=%v want none (b is stopping: its 800MB is already in usedMB=900 and about to be freed, not a load to reserve for)", evict)
	}
}

// TestBudgetSwapper_ShareExcludesReservation checks that the assumed
// per-eviction yield comes from the sampled usage only. Reserved memory
// belongs to models that have not loaded yet, so evicting a ready model
// cannot free any of it.
func TestBudgetSwapper_ShareExcludesReservation(t *testing.T) {
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
	s := newTestBudgetSwapper(1000, priority, processes, 900)
	s.recordLearned([]string{"c"}, 600)
	s.recordLearned([]string{"c"}, 600)
	s.recordLearned([]string{"c"}, 600)

	evict := s.EvictionFor("d", []string{"a", "b", "c"})
	if len(evict) != 2 || evict[0] != "a" || evict[1] != "b" {
		t.Fatalf("evict=%v want [a b] (300MB share from the 900MB sample, not 500MB from sample+reservation)", evict)
	}
}

// TestBudget_ObserveAndLearnRecordsFromSysMemSource checks that the
// unified-memory fallback still learns footprints. Its samples count every
// process on the host, so single observations can be noise; the median over
// the history (see TestBudgetSwapper_MedianRejectsOutlier) handles that.
func TestBudget_ObserveAndLearnRecordsFromSysMemSource(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady() // a ready model while VRAM reads 0 marks the host as unified

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	var usedMB atomic.Int64
	usedMB.Store(100)
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a, "b": b}, &usedMB)
	r.swapper.sampleVRAMUsedMB = func() (int64, time.Time, bool) { return 0, time.Now(), true }
	r.swapper.sampleSysMemUsedMB = func() (int64, time.Time, bool) { return usedMB.Load(), time.Now(), true }

	r.observeAndLearn() // baseline: only b resident

	a.setState(process.StateStarting)
	r.observeAndLearn()

	a.markReady()
	usedMB.Store(400) // could be the model, could be anything else on the host
	r.observeAndLearn()

	if hist := learnedHistory(r, "a"); len(hist) != 1 || hist[0] != 300 {
		t.Fatalf("learned[a]=%v want [300] (400-100 delta recorded even from a system-memory sample)", hist)
	}
}

// learnedHistory returns a copy of id's learned footprint history.
func learnedHistory(r *Budget, id string) []int64 {
	r.swapper.learnedMu.RLock()
	defer r.swapper.learnedMu.RUnlock()
	return append([]int64(nil), r.swapper.learned[id]...)
}

func TestBudget_ObserveAndLearnAttributesDelta(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	var usedMB atomic.Int64
	usedMB.Store(100)
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a}, &usedMB)

	r.observeAndLearn() // baseline, a not running

	a.setState(process.StateStarting)
	r.observeAndLearn()

	a.markReady()
	usedMB.Store(400)
	r.observeAndLearn() // a became ready, delta=300

	if hist := learnedHistory(r, "a"); len(hist) != 1 || hist[0] != 300 {
		t.Fatalf("learned[a]=%v want [300] (400-100 delta attributed to a)", hist)
	}
}

// TestBudget_ObserveAndLearnUsesPreLoadBaseline covers footprints learned too
// low: the weights are resident while the model is still starting, so the
// delta must be measured from before the load began, not from the last tick
// before it turned ready.
func TestBudget_ObserveAndLearnUsesPreLoadBaseline(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 100000}}}},
	}
	var usedMB atomic.Int64
	usedMB.Store(1000)
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a}, &usedMB)

	r.observeAndLearn() // a not running

	a.setState(process.StateStarting)
	usedMB.Store(6000) // weights loading
	r.observeAndLearn()
	usedMB.Store(8500) // weights loaded, still warming up
	r.observeAndLearn()

	a.markReady()
	usedMB.Store(9000)
	r.observeAndLearn()

	if hist := learnedHistory(r, "a"); len(hist) != 1 || hist[0] != 8000 {
		t.Fatalf("learned[a]=%v want [8000] (9000 now minus 1000 before the load began, not the last 500MB step)", hist)
	}
}

// TestBudget_ObserveAndLearnSkipsWhenModelStoppedDuringLoad checks that a
// model stopping while another loads disables learning: its freed memory
// would hide part of the loading model's footprint.
func TestBudget_ObserveAndLearnSkipsWhenModelStoppedDuringLoad(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 100000}}}},
	}
	var usedMB atomic.Int64
	usedMB.Store(5000)
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 1}, map[string]process.Process{"a": a, "b": b}, &usedMB)

	r.observeAndLearn() // baseline: b resident

	a.setState(process.StateStarting)
	b.setState(process.StateStopped) // b evicted to make room
	r.observeAndLearn()

	a.markReady()
	usedMB.Store(6000) // a's footprint minus b's freed memory
	r.observeAndLearn()

	if hist := learnedHistory(r, "a"); len(hist) != 0 {
		t.Fatalf("learned[a]=%v want none (b stopped during a's load)", hist)
	}
}

func TestBudget_ObserveAndLearnSplitsDeltaAcrossConcurrentLoads(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	b := newFakeProcess("b")
	b.testPid = 101

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	var usedMB atomic.Int64
	usedMB.Store(100)
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 1}, map[string]process.Process{"a": a, "b": b}, &usedMB)

	r.observeAndLearn()

	a.setState(process.StateStarting)
	b.setState(process.StateStarting)
	usedMB.Store(400)
	r.observeAndLearn()

	a.markReady() // a ready, b still loading: wait for both
	usedMB.Store(600)
	r.observeAndLearn()

	b.markReady()
	usedMB.Store(700) // 600 delta from the shared pre-load baseline, split evenly
	r.observeAndLearn()

	histA, histB := learnedHistory(r, "a"), learnedHistory(r, "b")
	if len(histA) != 1 || histA[0] != 300 || len(histB) != 1 || histB[0] != 300 {
		t.Fatalf("learned a=%v b=%v want a=[300] b=[300] (600 delta split across 2 concurrent loads)", histA, histB)
	}
}

func TestBudget_ObserveAndLearnSkipsNonPositiveDelta(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	var usedMB atomic.Int64
	usedMB.Store(500)
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a}, &usedMB)

	r.observeAndLearn()

	a.setState(process.StateStarting)
	r.observeAndLearn()

	a.markReady()
	usedMB.Store(300) // usage fell (something outside llama-swap freed memory)
	r.observeAndLearn()

	if hist := learnedHistory(r, "a"); len(hist) != 0 {
		t.Fatalf("learned[a]=%v should not be recorded from a non-positive delta", hist)
	}
}

// TestBudget_ObserveAndLearnWaitsForSampleAfterReady checks that a sample
// taken before the model became ready is not used to close the load: it may
// miss the end of the load.
func TestBudget_ObserveAndLearnWaitsForSampleAfterReady(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 100000}}}},
	}
	r := newTestBudget(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a}, 0)
	var usedMB atomic.Int64
	var sampledAt atomic.Int64
	r.swapper.sampleVRAMUsedMB = func() (int64, time.Time, bool) {
		return usedMB.Load(), time.Unix(0, sampledAt.Load()), true
	}
	sample := func(mb int64) {
		usedMB.Store(mb)
		sampledAt.Store(time.Now().UnixNano())
	}

	sample(1000)
	r.observeAndLearn()

	a.setState(process.StateStarting)
	sample(5000) // taken before a became ready
	r.observeAndLearn()

	a.markReady()
	r.observeAndLearn() // same stale sample
	if hist := learnedHistory(r, "a"); len(hist) != 0 {
		t.Fatalf("learned[a]=%v want none from a sample taken before a became ready", hist)
	}

	sample(9000)
	r.observeAndLearn()
	if hist := learnedHistory(r, "a"); len(hist) != 1 || hist[0] != 8000 {
		t.Fatalf("learned[a]=%v want [8000] from the first sample after a became ready", hist)
	}
}

// TestBudget_EnforceBudgetOnceResumesWhenNewLoadExplainsTheRise covers
// the false-pause bug: usage rising since the last eviction does not mean the
// eviction failed if a concurrent load finished in between and explains the
// rise on its own. Pausing in that case disables protection for as long as
// the burst continues — exactly when it's needed most.
func TestBudget_EnforceBudgetOnceResumesWhenNewLoadExplainsTheRise(t *testing.T) {
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
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	var usedMB atomic.Int64
	usedMB.Store(1200)
	processes := map[string]process.Process{"a": a, "b": b, "c": c}
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 2, "c": 3}, processes, &usedMB)

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

// TestBudget_NextCheckIntervalTightensWhileLoading checks that the
// periodic check switches to the tighter interval whenever any model is
// mid-load, and relaxes back once nothing is.
func TestBudget_NextCheckIntervalTightensWhileLoading(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	r := newTestBudget(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a}, 100)

	if got := r.nextCheckInterval(); got != budgetCheckInterval {
		t.Errorf("nextCheckInterval=%v want %v (nothing loading)", got, budgetCheckInterval)
	}

	b := newFakeProcess("b")
	b.testPid = 101
	b.setState(process.StateStarting)
	r.processes["b"] = b

	if got := r.nextCheckInterval(); got != budgetActiveLoadCheckInterval {
		t.Errorf("nextCheckInterval=%v want %v (b is loading)", got, budgetActiveLoadCheckInterval)
	}

	b.markReady()
	if got := r.nextCheckInterval(); got != budgetCheckInterval {
		t.Errorf("nextCheckInterval=%v want %v (b finished loading)", got, budgetCheckInterval)
	}
}

// TestBudgetSwapper_GracePeriodProtectsJustLoadedModel reproduces the
// reported bug directly: a low-priority model that just finished loading
// must not be the first thing periodic eviction reaches for, even though
// priority alone would normally put it first — it may still be serving the
// very request that just loaded it, and periodic eviction (via
// Budget.Unload) does not check in-flight requests before stopping a
// process, unlike an admission-time swap.
func TestBudgetSwapper_GracePeriodProtectsJustLoadedModel(t *testing.T) {
	nomic := newFakeProcess("nomic")
	nomic.testPid = 100
	nomic.markReady()
	dflash := newFakeProcess("dflash")
	dflash.testPid = 101
	dflash.markReady()
	mtp := newFakeProcess("mtp")
	mtp.testPid = 102
	mtp.markReady() // just finished loading

	processes := map[string]process.Process{"nomic": nomic, "dflash": dflash, "mtp": mtp}
	// Mirrors the reported config: mtp is the lowest priority of the three,
	// which would normally make it the first eviction target.
	priority := map[string]int{"nomic": 10, "dflash": 10, "mtp": 1}
	s := newTestBudgetSwapper(1000, priority, processes, 1500)
	s.recordReadySince([]string{"mtp"}, time.Now()) // mtp just became ready

	evict := s.pickEvictions([]string{"nomic", "dflash", "mtp"}, 1500, 0, 3)
	if len(evict) == 0 || evict[0] == "mtp" {
		t.Fatalf("evict=%v want mtp deferred to last despite its lower priority (it just loaded)", evict)
	}
}

// TestBudgetSwapper_GracePeriodExpires checks the other half: once
// budgetEvictionGracePeriod has actually elapsed, priority ordering
// applies normally again.
func TestBudgetSwapper_GracePeriodExpires(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()

	processes := map[string]process.Process{"a": a, "b": b}
	priority := map[string]int{"a": 10, "b": 1}
	s := newTestBudgetSwapper(1000, priority, processes, 1500)
	s.recordReadySince([]string{"b"}, time.Now().Add(-time.Hour)) // long past the grace period

	evict := s.pickEvictions([]string{"a", "b"}, 1500, 0, 2)
	if len(evict) == 0 || evict[0] != "b" {
		t.Fatalf("evict=%v want b first (its grace period has long since expired, so priority applies normally)", evict)
	}
}

// TestBudgetSwapper_GracePeriodStillEvictsIfNothingElseAvailable
// checks the fallback: if every over-budget candidate is inside its grace
// period, eviction still proceeds into them rather than never reclaiming
// memory at all.
func TestBudgetSwapper_GracePeriodStillEvictsIfNothingElseAvailable(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()

	processes := map[string]process.Process{"a": a}
	s := newTestBudgetSwapper(1000, map[string]int{"a": 1}, processes, 1500)
	s.recordReadySince([]string{"a"}, time.Now())

	evict := s.pickEvictions([]string{"a"}, 1500, 0, 1)
	if len(evict) != 1 || evict[0] != "a" {
		t.Fatalf("evict=%v want [a] (only candidate, grace period must not block eviction entirely)", evict)
	}
}

// TestBudget_EnforceBudgetOnceNeverEvictsBusyModel reproduces the live
// incident directly: a low-priority model handling an in-flight request must
// survive periodic eviction even once its own load-time grace period has
// long expired, because periodic eviction (via Unload/StopProcesses) has no
// in-flight check of its own — unlike an admission-time swap, which the FIFO
// scheduler queues behind instead of evicting into.
func TestBudget_EnforceBudgetOnceNeverEvictsBusyModel(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.autoReady = true
	a.serveBlock = make(chan struct{})
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	// a is the lowest priority, normally the first eviction target.
	r := newTestBudget(t, conf, map[string]int{"a": 1, "b": 10}, map[string]process.Process{"a": a, "b": b}, 1500)

	// a's grace period from loading has long since expired: protection here
	// must come from it being busy right now, not from having just loaded.
	r.swapper.recordReadySince([]string{"a"}, time.Now().Add(-time.Hour))

	done := make(chan struct{})
	go func() {
		r.ServeHTTP(httptest.NewRecorder(), newStreamRequest("a"))
		close(done)
	}()
	waitSignal(t, a.serveStarted, "a request start")
	waitProcessed(t, r.testProcessed, 1)

	r.enforceBudgetOnce()
	if got := a.stopCalls.Load(); got != 0 {
		t.Errorf("a.stopCalls=%d want 0 (busy serving a request, must not be evicted)", got)
	}
	if got := b.stopCalls.Load(); got != 1 {
		t.Errorf("b.stopCalls=%d want 1 (a is busy and excluded, so b is the only eligible candidate despite its higher priority)", got)
	}

	close(a.serveBlock)
	waitSignal(t, done, "a request finish")
}

// TestBudget_NextCheckIntervalNotShorterThanSampling checks that the
// active-load interval is clamped to the perf monitor's sampling interval:
// checking more often would only re-read the same sample.
func TestBudget_NextCheckIntervalNotShorterThanSampling(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.setState(process.StateStarting)

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	r := newTestBudget(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a}, 100)
	r.sampleEvery = func() time.Duration { return 5 * time.Second }

	if got := r.nextCheckInterval(); got != 5*time.Second {
		t.Errorf("nextCheckInterval=%v want 5s (perf monitor samples every 5s)", got)
	}
}

// TestBudget_EnforceBudgetOnceIgnoresStaleSampleAfterEviction covers the
// false "freed nothing" pause: a check that re-reads the sample taken before
// the eviction must neither pause nor evict again, but wait for a fresh
// sample.
func TestBudget_EnforceBudgetOnceIgnoresStaleSampleAfterEviction(t *testing.T) {
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
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	processes := map[string]process.Process{"a": a, "b": b, "c": c}
	r := newTestBudget(t, conf, map[string]int{"a": 1, "b": 2, "c": 3}, processes, 0)
	var usedMB atomic.Int64
	var sampledAt atomic.Int64
	r.swapper.sampleVRAMUsedMB = func() (int64, time.Time, bool) {
		return usedMB.Load(), time.Unix(0, sampledAt.Load()), true
	}
	sample := func(mb int64) {
		usedMB.Store(mb)
		sampledAt.Store(time.Now().UnixNano())
	}

	sample(1500)
	r.enforceBudgetOnce()
	if got := a.stopCalls.Load(); got != 1 {
		t.Fatalf("a.stopCalls=%d want 1 on the first check", got)
	}

	// No new sample yet: the monitor has not sampled since the eviction.
	r.enforceBudgetOnce()
	if r.warnedEvictFreedNothing {
		t.Errorf("paused on a sample taken before the eviction")
	}
	if got := b.stopCalls.Load(); got != 0 {
		t.Errorf("b.stopCalls=%d want 0 (no fresh sample to decide on)", got)
	}

	// A fresh sample shows the eviction helped but usage is still over.
	sample(1200)
	r.enforceBudgetOnce()
	if got := b.stopCalls.Load(); got != 1 {
		t.Errorf("b.stopCalls=%d want 1 (fresh sample, still over budget)", got)
	}
	if got := c.stopCalls.Load(); got != 0 {
		t.Errorf("c.stopCalls=%d want 0", got)
	}
}

// TestBudgetSwapper_IdleDiscreteGPUIsNotUnified checks that a discrete GPU
// reading 0MB with nothing loaded is budgeted as VRAM, not treated as unified
// memory, and that once VRAM has read above 0 it stays the source.
func TestBudgetSwapper_IdleDiscreteGPUIsNotUnified(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100

	var logs bytes.Buffer
	var vramMB atomic.Int64
	s := &budgetSwapper{
		limitMB:   1000,
		processes: map[string]process.Process{"a": a},
		logger:    logmon.NewWriter(&logs),
		sampleVRAMUsedMB: func() (int64, time.Time, bool) {
			return vramMB.Load(), time.Now(), true
		},
		sampleSysMemUsedMB: func() (int64, time.Time, bool) { return 4000, time.Now(), true },
	}

	if smp := s.usedMemoryMB(); smp.src != sourceVRAM || smp.mb != 0 {
		t.Fatalf("idle sample=%+v want VRAM 0 (nothing loaded)", smp)
	}

	a.markReady()
	vramMB.Store(3000)
	if smp := s.usedMemoryMB(); smp.src != sourceVRAM || smp.mb != 3000 {
		t.Fatalf("loaded sample=%+v want VRAM 3000", smp)
	}

	// A later 0 reading on a GPU known to report usage stays VRAM.
	vramMB.Store(0)
	if smp := s.usedMemoryMB(); smp.src != sourceVRAM || smp.mb != 0 {
		t.Fatalf("sample=%+v want VRAM 0 (discrete GPU already seen above 0)", smp)
	}
	if bytes.Contains(logs.Bytes(), []byte("unified")) {
		t.Errorf("unified-memory warning logged for a discrete GPU: %s", logs.String())
	}
}

// TestBudget_ObserveAndLearnIdleDiscreteGPUAtZero covers a fake footprint
// from comparing sources: a discrete GPU reading 0 with nothing loaded must
// not put system memory in the baseline.
func TestBudget_ObserveAndLearnIdleDiscreteGPUAtZero(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 100000}}}},
	}
	var vramMB atomic.Int64
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a}, &vramMB)
	r.swapper.sampleSysMemUsedMB = func() (int64, time.Time, bool) { return 2000, time.Now(), true }

	r.observeAndLearn() // idle GPU at 0

	a.setState(process.StateStarting)
	vramMB.Store(5000)
	r.observeAndLearn()

	a.markReady()
	vramMB.Store(6000)
	r.observeAndLearn()

	if hist := learnedHistory(r, "a"); len(hist) != 1 || hist[0] != 6000 {
		t.Fatalf("learned[a]=%v want [6000] (VRAM 0 -> 6000, not system memory 2000 -> VRAM 6000)", hist)
	}
}

// TestBudget_ObserveAndLearnFirstLoadOnUnifiedHost checks that the first load
// on an empty unified-memory host is learned: the baseline is taken while VRAM
// reads 0 and the host is unclassified, and the closing sample is system
// memory once the model is ready.
func TestBudget_ObserveAndLearnFirstLoadOnUnifiedHost(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 100000}}}},
	}
	var vramMB, sysMB atomic.Int64
	sysMB.Store(2000)
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a}, &vramMB)
	r.swapper.sampleSysMemUsedMB = func() (int64, time.Time, bool) { return sysMB.Load(), time.Now(), true }

	r.observeAndLearn() // empty host, VRAM 0, unclassified

	a.setState(process.StateStarting)
	sysMB.Store(6000)
	r.observeAndLearn()

	a.markReady()
	sysMB.Store(8000) // VRAM still 0: host is unified
	r.observeAndLearn()

	if hist := learnedHistory(r, "a"); len(hist) != 1 || hist[0] != 6000 {
		t.Fatalf("learned[a]=%v want [6000] (system memory 2000 -> 8000)", hist)
	}
}

// TestBudget_ObserveAndLearnSkipsAcrossSourceChange checks that no footprint
// is learned when the baseline and the closing sample come from different
// memory pools.
func TestBudget_ObserveAndLearnSkipsAcrossSourceChange(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 100000}}}},
	}
	var vramMB atomic.Int64
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1}, map[string]process.Process{"a": a, "b": b}, &vramMB)
	r.swapper.sampleSysMemUsedMB = func() (int64, time.Time, bool) { return 2000, time.Now(), true }

	r.observeAndLearn() // VRAM 0 with b ready: system memory baseline

	a.setState(process.StateStarting)
	r.observeAndLearn()

	a.markReady()
	vramMB.Store(9000) // VRAM starts reporting
	r.observeAndLearn()

	if hist := learnedHistory(r, "a"); len(hist) != 0 {
		t.Fatalf("learned[a]=%v want none (baseline was system memory, sample is VRAM)", hist)
	}
}

// TestBudget_EnforceBudgetOnceResetsAcrossSourceChange checks that a usage
// comparison against the last eviction is skipped when the memory source
// changed, rather than pausing on an apples-to-oranges comparison.
func TestBudget_EnforceBudgetOnceResetsAcrossSourceChange(t *testing.T) {
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
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	var vramMB atomic.Int64
	processes := map[string]process.Process{"a": a, "b": b, "c": c}
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 2, "c": 3}, processes, &vramMB)
	r.swapper.sampleSysMemUsedMB = func() (int64, time.Time, bool) { return 1200, time.Now(), true }

	r.enforceBudgetOnce() // VRAM 0 with models ready: system memory 1200
	if got := a.stopCalls.Load(); got != 1 {
		t.Fatalf("a.stopCalls=%d want 1", got)
	}

	vramMB.Store(1500) // VRAM starts reporting, higher than the sysmem figure
	r.enforceBudgetOnce()
	if r.warnedEvictFreedNothing {
		t.Errorf("paused comparing a VRAM sample with a system-memory one")
	}
	if got := b.stopCalls.Load(); got != 1 {
		t.Errorf("b.stopCalls=%d want 1 (over budget on the new source)", got)
	}
}

// TestBudget_EnforceBudgetOnceSkipsModelWithQueuedRequest covers clients
// erroring out: an idle model with a request queued for it must not be
// evicted, since the eviction would fail that request.
func TestBudget_EnforceBudgetOnceSkipsModelWithQueuedRequest(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()
	b.serveBlock = make(chan struct{})
	c := newFakeProcess("c")
	c.testPid = 102
	c.markReady()

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	processes := map[string]process.Process{"a": a, "b": b, "c": c}
	r := newTestBudget(t, conf, map[string]int{"a": 0, "b": 1, "c": 5}, processes, 1500)

	// b busy serving a request.
	bDone := make(chan struct{})
	go func() {
		r.ServeHTTP(httptest.NewRecorder(), newRequest("b"))
		close(bDone)
	}()
	waitSignal(t, b.serveStarted, "b request start")
	waitProcessed(t, r.testProcessed, 1)

	// A request for a would evict busy b, so it queues.
	aResp := httptest.NewRecorder()
	aDone := make(chan struct{})
	go func() {
		r.ServeHTTP(aResp, newRequest("a"))
		close(aDone)
	}()
	waitProcessed(t, r.testProcessed, 1)

	// a comes up out of band while its request is still queued.
	a.markReady()

	r.enforceBudgetOnce()
	if got := a.stopCalls.Load(); got != 0 {
		t.Errorf("a.stopCalls=%d want 0 (a request is queued for it)", got)
	}
	if got := c.stopCalls.Load(); got != 1 {
		t.Errorf("c.stopCalls=%d want 1 (the only idle model)", got)
	}

	close(b.serveBlock)
	waitSignal(t, bDone, "b request finish")
	waitSignal(t, aDone, "a request finish")
	if aResp.Code != http.StatusOK {
		t.Errorf("a status=%d body=%q want 200", aResp.Code, aResp.Body.String())
	}
}

// TestBudget_EvictionDoesNotBlockRouting checks that a periodic eviction
// stops its model off the run loop: other models keep serving while it
// stops, and a request for the model being evicted waits for the stop and
// then reloads it instead of being handed the dying process.
func TestBudget_EvictionDoesNotBlockRouting(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	a.autoReady = true
	a.stopBlock = make(chan struct{})
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()
	c := newFakeProcess("c")
	c.testPid = 102
	c.markReady()

	conf := config.Config{
		HealthCheckTimeout: 5,
		Routing:            config.RoutingConfig{Router: config.RouterConfig{Settings: config.RouterSettings{Budget: &config.BudgetConfig{LimitMB: 1000}}}},
	}
	processes := map[string]process.Process{"a": a, "b": b, "c": c}
	var usedMB atomic.Int64
	usedMB.Store(1200)
	r := newTestBudgetWithUsage(t, conf, map[string]int{"a": 1, "b": 10, "c": 10}, processes, &usedMB)

	enforced := make(chan struct{})
	go func() {
		r.enforceBudgetOnce()
		close(enforced)
	}()
	waitSignal(t, a.stopStarted, "a stop start")

	// The run loop is free while a stops.
	w := httptest.NewRecorder()
	r.ServeHTTP(w, newRequest("b"))
	if w.Code != http.StatusOK {
		t.Fatalf("b status=%d want 200 while a is stopping", w.Code)
	}
	waitProcessed(t, r.testProcessed, 1)

	// A request for a queues until the stop finishes.
	usedMB.Store(500)
	aResp := httptest.NewRecorder()
	aDone := make(chan struct{})
	go func() {
		r.ServeHTTP(aResp, newRequest("a"))
		close(aDone)
	}()
	waitProcessed(t, r.testProcessed, 1)
	if got := a.serveCalls.Load(); got != 0 {
		t.Fatalf("a.serveCalls=%d want 0 (a is being evicted)", got)
	}

	close(a.stopBlock)
	waitSignal(t, enforced, "enforceBudgetOnce return")
	waitSignal(t, aDone, "a request finish")
	if aResp.Code != http.StatusOK {
		t.Errorf("a status=%d body=%q want 200 after reload", aResp.Code, aResp.Body.String())
	}
	if got := a.runCalls.Load(); got != 1 {
		t.Errorf("a.runCalls=%d want 1 (reloaded after the eviction)", got)
	}
}

// TestBudgetSwapper_OnSwapStartLogsDecidedEvictSet checks that the logged
// eviction is the one the scheduler acted on, not a fresh decision from a
// newer sample.
func TestBudgetSwapper_OnSwapStartLogsDecidedEvictSet(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()

	var logs bytes.Buffer
	var usedMB atomic.Int64
	usedMB.Store(1500)
	s := newTestBudgetSwapper(1000, map[string]int{"a": 1, "b": 10}, map[string]process.Process{"a": a, "b": b}, 0)
	s.logger = logmon.NewWriter(&logs)
	s.sampleVRAMUsedMB = func() (int64, time.Time, bool) { return usedMB.Load(), time.Now(), true }

	running := []string{"a", "b"}
	evict := s.EvictionFor("c", running)
	usedMB.Store(100) // a newer sample would decide on no eviction
	s.OnSwapStart("c", running, evict)

	if want := "budget: model=c usedMB=1500 reservedMB=0 limitMB=1000 evict=[a]"; !bytes.Contains(logs.Bytes(), []byte(want)) {
		t.Errorf("log=%q want it to contain %q", logs.String(), want)
	}
}
