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
