package router

import (
	"context"
	"fmt"
	"slices"
	"sort"
	"sync"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/perf"
	"github.com/mostlygeek/llama-swap/internal/process"
)

// memoryBudgetCheckInterval is how often enforceBudgetPeriodically re-samples
// usage independent of any swap. See that method's doc comment for why this
// exists alongside the swap-start check in EvictionFor.
const memoryBudgetCheckInterval = 15 * time.Second

type MemoryBudget struct {
	*baseRouter
	swapper *memoryBudgetSwapper
}

func NewMemoryBudget(conf config.Config, proxylog, upstreamlog *logmon.Monitor, perfMon *perf.Monitor) (*MemoryBudget, error) {
	mb := conf.Routing.Router.Settings.MemoryBudget
	if mb == nil {
		return nil, fmt.Errorf("memoryBudget router requires a memoryBudget configuration")
	}
	if perfMon == nil {
		return nil, fmt.Errorf("memoryBudget router requires performance monitoring (GPU/VRAM tracking) to be enabled")
	}

	processes := make(map[string]process.Process, len(conf.Models))
	swapper := &memoryBudgetSwapper{
		limitMB:            mb.Limit.MB(),
		priority:           make(map[string]int, len(mb.Models)),
		processes:          processes,
		logger:             proxylog,
		sampleVRAMUsedMB:   perfMon.VRAMUsedMB,
		sampleSysMemUsedMB: perfMon.SysMemUsedMB,
	}
	for modelID, entry := range mb.Models {
		swapper.priority[modelID] = entry.Priority
	}

	base, err := newBaseRouter("memoryBudget", conf, processes, proxylog, swapper)
	if err != nil {
		return nil, fmt.Errorf("creating base router: %w", err)
	}

	// Build a process for every model, same as Matrix: any model can run
	// alone even if it has no explicit priority entry (default priority 0).
	for mid, modelCfg := range conf.Models {
		procLog := logmon.NewWriter(upstreamlog)
		p, err := process.New(base.procCtx, mid, modelCfg, procLog, proxylog)
		if err != nil {
			base.shutdownFn()
			base.procCancel()
			return nil, fmt.Errorf("creating process for %q: %w", mid, err)
		}
		processes[mid] = p
	}

	r := &MemoryBudget{baseRouter: base, swapper: swapper}
	go base.run()
	go r.enforceBudgetPeriodically(base.procCtx)
	return r, nil
}

// enforceBudgetPeriodically re-checks usage on a timer and evicts even when no
// new model is loading.
//
// EvictionFor only runs at swap-start, and it samples usage *before* the
// incoming model finishes loading — so the model that actually tips the host
// over budget is never itself weighed against the limit. If nothing else
// tries to load afterwards, there is no other trigger that would notice, and
// the host just sits over budget (see the memoryBudgetSwapper doc comment for
// why per-model footprint can't be predicted ahead of a load to fix this at
// admission time instead). This closes that gap by checking independent of
// swap events.
func (mb *MemoryBudget) enforceBudgetPeriodically(ctx context.Context) {
	ticker := time.NewTicker(memoryBudgetCheckInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			mb.enforceBudgetOnce()
		}
	}
}

func (mb *MemoryBudget) enforceBudgetOnce() {
	usedMB, ok := mb.swapper.usedMemoryMB()
	if !ok {
		return
	}
	if usedMB <= mb.swapper.limitMB {
		mb.logger.Debugf("memoryBudget: periodic check usedMB=%d limitMB=%d (under budget)", usedMB, mb.swapper.limitMB)
		return
	}

	states := mb.RunningModels()
	running := make([]string, 0, len(states))
	ready := make([]string, 0, len(states))
	for id, st := range states {
		running = append(running, id)
		if st == process.StateReady {
			ready = append(ready, id)
		}
	}
	if len(ready) == 0 {
		// Over budget with nothing evictable yet (e.g. the only running
		// model is still starting) — nothing to do until the next tick.
		mb.logger.Debugf("memoryBudget: periodic check usedMB=%d limitMB=%d over budget but no ready model to evict yet", usedMB, mb.swapper.limitMB)
		return
	}

	evict := mb.swapper.pickEvictions(ready, usedMB, len(running))
	if len(evict) == 0 {
		return
	}
	mb.logger.Infof("memoryBudget: periodic check usedMB=%d limitMB=%d evicting=%v (over budget with no new model loading to trigger EvictionFor)", usedMB, mb.swapper.limitMB, evict)
	mb.Unload(0, evict...)
}

// memoryBudgetSwapper decides evictions by keeping total VRAM used (summed
// across all GPUs, sampled live from the host's GPU monitor) under a
// configured limit, evicting lowest-priority (ties broken by
// least-recently-used) models first, and continuing through higher-priority
// models too if that's what it takes to fit the requested model.
//
// On hosts with unified CPU/GPU memory, where GPU tools report 0MB VRAM used,
// the budget is applied to system memory used instead (see usedMemoryMB).
//
// VRAM usage is not attributed per model: the GPU monitoring tools this
// relies on (nvidia-smi, rocm-smi, LACT, ...) report total VRAM used across
// all processes on the GPU, not a per-model breakdown, and per-process
// attribution isn't available consistently across vendors. So instead of
// tracking each model's individual footprint, eviction treats the current
// total as a shared pool: each evicted model is assumed to free an equal
// share of it.
type memoryBudgetSwapper struct {
	limitMB   int64
	priority  map[string]int // model ID -> priority, default 0
	processes map[string]process.Process
	logger    *logmon.Monitor

	// sampleVRAMUsedMB returns total VRAM used in MB across all GPUs, and
	// false if no sample is available yet. Defaults to perfMon.VRAMUsedMB;
	// tests override it to avoid depending on real GPU monitoring tools.
	sampleVRAMUsedMB func() (int64, bool)

	// sampleSysMemUsedMB returns system memory used in MB, and false if no
	// sample is available yet. Used as the budget signal on unified-memory
	// hosts, where VRAM reads 0. Defaults to perfMon.SysMemUsedMB.
	sampleSysMemUsedMB func() (int64, bool)

	warnUnified sync.Once
}

// usedMemoryMB samples memory used by the pool the budget applies to.
//
// Normally that is total VRAM across all GPUs. Hosts with unified CPU/GPU
// memory (NVIDIA Grace-Blackwell GB10, for example) have no separate
// framebuffer, so nvidia-smi reports memory.used as 0 even while models are
// resident; budgeting against a permanent 0 would silently disable eviction.
// There the pool is system RAM, so fall back to system memory used (the same
// figure `free` reports as used) and note the switch once.
//
// ok is false when neither source has produced a sample yet.
func (s *memoryBudgetSwapper) usedMemoryMB() (int64, bool) {
	if usedMB, ok := s.sampleVRAMUsedMB(); ok && usedMB > 0 {
		return usedMB, true
	}
	if s.sampleSysMemUsedMB == nil {
		return 0, false
	}
	usedMB, ok := s.sampleSysMemUsedMB()
	if !ok {
		return 0, false
	}
	s.warnUnified.Do(func() {
		s.logger.Warnf("memoryBudget: GPU monitoring reports 0MB VRAM used, budgeting against system memory used (%dMB) instead. Expected on hosts with unified CPU/GPU memory (for example NVIDIA Grace-Blackwell GB10); note that system memory used counts everything on the host, not only models.", usedMB)
	})
	return usedMB, true
}

func (s *memoryBudgetSwapper) EvictionFor(target string, running []string) []string {
	if slices.Contains(running, target) || len(running) == 0 {
		return nil
	}

	usedMB, ok := s.usedMemoryMB()
	if !ok || usedMB <= s.limitMB {
		return nil
	}

	return s.pickEvictions(running, usedMB, len(running))
}

// pickEvictions greedily selects the lowest-priority (ties broken by
// least-recently-used) models from candidates until usedMB, decremented by an
// equal share per eviction, would drop to or below the limit. shareDivisor is
// the size of the pool the share is computed against — usually len(candidates),
// but callers evicting from a subset of the running set (e.g. periodic
// enforcement skipping still-starting processes) pass the full running count
// so the assumed per-model share stays accurate.
//
// VRAM isn't attributed per model (see the memoryBudgetSwapper doc comment),
// so each eviction is assumed to free an equal share of the total currently
// used.
func (s *memoryBudgetSwapper) pickEvictions(candidates []string, usedMB int64, shareDivisor int) []string {
	if shareDivisor == 0 {
		return nil
	}

	sorted := make([]string, len(candidates))
	copy(sorted, candidates)
	sort.Slice(sorted, func(i, j int) bool {
		pi, pj := s.priority[sorted[i]], s.priority[sorted[j]]
		if pi != pj {
			return pi < pj
		}
		return s.lastUseOf(sorted[i]).Before(s.lastUseOf(sorted[j]))
	})

	shareMB := usedMB / int64(shareDivisor)

	var evict []string
	for _, m := range sorted {
		if usedMB <= s.limitMB {
			break
		}
		evict = append(evict, m)
		usedMB -= shareMB
	}
	return evict
}

func (s *memoryBudgetSwapper) OnSwapStart(target string, running []string) {
	evict := s.EvictionFor(target, running)
	usedMB, ok := s.usedMemoryMB()
	switch {
	case len(evict) > 0:
		s.logger.Infof("memoryBudget: model=%s usedMB=%d limitMB=%d evict=%v", target, usedMB, s.limitMB, evict)
	case len(running) == 0:
		s.logger.Infof("memoryBudget: model=%s starting (no models running)", target)
	case !ok:
		s.logger.Debugf("memoryBudget: model=%s starting (no memory sample available yet)", target)
	default:
		s.logger.Debugf("memoryBudget: model=%s fits without eviction (usedMB=%d limitMB=%d)", target, usedMB, s.limitMB)
	}
}

func (s *memoryBudgetSwapper) lastUseOf(modelID string) time.Time {
	if p, ok := s.processes[modelID]; ok {
		return p.LastUse()
	}
	return time.Time{}
}
