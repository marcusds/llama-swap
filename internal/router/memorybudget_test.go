package router

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/process"
)

// newTestMemoryBudgetSwapper builds a memoryBudgetSwapper with a stubbed RSS
// sampler so tests can set memory usage deterministically without depending
// on real OS processes.
func newTestMemoryBudgetSwapper(limitMB int64, priority map[string]int, processes map[string]process.Process, rss map[string]int64) *memoryBudgetSwapper {
	return &memoryBudgetSwapper{
		limitMB:   limitMB,
		priority:  priority,
		memoryMB:  make(map[string]int64),
		processes: processes,
		logger:    logmon.NewWriter(io.Discard),
		sampleRSS: func(pid int) (int64, error) {
			for id, p := range processes {
				if fp, ok := p.(*fakeProcess); ok && fp.testPid == pid {
					return rss[id], nil
				}
			}
			return 0, fmt.Errorf("no rss stubbed for pid %d", pid)
		},
	}
}

func TestMemoryBudgetSwapper_AlreadyRunning(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	processes := map[string]process.Process{"a": a}
	s := newTestMemoryBudgetSwapper(1000, nil, processes, map[string]int64{"a": 800})

	evict := s.EvictionFor("a", []string{"a"})
	if len(evict) != 0 {
		t.Errorf("evict=%v want none (already running)", evict)
	}
}

func TestMemoryBudgetSwapper_FitsWithoutEviction(t *testing.T) {
	a := newFakeProcess("a")
	a.testPid = 100
	a.markReady()
	b := newFakeProcess("b")
	b.testPid = 101
	b.markReady()
	processes := map[string]process.Process{"a": a, "b": b}
	s := newTestMemoryBudgetSwapper(1000, nil, processes, map[string]int64{"a": 300, "b": 300})

	evict := s.EvictionFor("b", []string{"a"})
	if len(evict) != 0 {
		t.Errorf("evict=%v want none (fits under limit)", evict)
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
	rss := map[string]int64{"a": 500, "b": 500, "c": 500}
	s := newTestMemoryBudgetSwapper(1000, priority, processes, rss)

	// c is loaded, requesting a new 500MB model needs 1000MB total: only c
	// (lowest priority) must be evicted to fit under the 1000MB cap.
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
	rss := map[string]int64{"a": 900, "b": 900}
	s := newTestMemoryBudgetSwapper(1000, priority, processes, rss)
	// Simulate c having run before, so its last-measured memory (900MB) is
	// known ahead of this decision instead of defaulting to 0.
	s.memoryMB["c"] = 900

	// requesting a 900MB model with only 1000MB total: evicting the
	// lowest-priority model (b) alone isn't enough (900+900 > 1000 still),
	// so eviction must proceed through a too.
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
	rss := map[string]int64{"a": 600, "b": 600}
	s := newTestMemoryBudgetSwapper(1000, priority, processes, rss)

	// Equal priority: the least-recently-used (a) is evicted first.
	evict := s.EvictionFor("c", []string{"a", "b"})
	if len(evict) != 1 || evict[0] != "a" {
		t.Fatalf("evict=%v want [a] (older lastUse)", evict)
	}
}

// newTestMemoryBudget builds a MemoryBudget router from supplied processes,
// bypassing NewMemoryBudget's call to process.New, mirroring newTestMatrix.
func newTestMemoryBudget(t *testing.T, conf config.Config, priority map[string]int, processes map[string]process.Process, rss map[string]int64) *MemoryBudget {
	t.Helper()
	swapper := newTestMemoryBudgetSwapper(conf.MemoryBudget.Limit.MB(), priority, processes, rss)
	// Seed memoryMB with every rss entry up front, as if each model had
	// already been observed running once. sampleAll only refreshes entries
	// for the models currently in the running set, so without this seed the
	// not-yet-started target's own memory need would default to 0.
	for id, mb := range rss {
		swapper.memoryMB[id] = mb
	}
	base, err := newBaseRouter("memoryBudget", conf, processes, swapper.logger, swapper)
	if err != nil {
		t.Fatalf("newBaseRouter: %v", err)
	}
	base.testProcessed = make(chan struct{}, 64)
	r := &MemoryBudget{baseRouter: base}
	go base.run()
	t.Cleanup(func() {
		if !r.shuttingDown.Load() {
			_ = r.Shutdown(time.Second)
		}
	})
	return r
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
	rss := map[string]int64{"a": 900, "b": 900}

	r := newTestMemoryBudget(t, conf, priority, map[string]process.Process{"a": a, "b": b}, rss)

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
