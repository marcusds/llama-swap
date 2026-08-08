package router

import (
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

type MemoryBudget struct {
	*baseRouter
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

	r := &MemoryBudget{baseRouter: base}
	go base.run()
	return r, nil
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

	candidates := make([]string, len(running))
	copy(candidates, running)
	sort.Slice(candidates, func(i, j int) bool {
		pi, pj := s.priority[candidates[i]], s.priority[candidates[j]]
		if pi != pj {
			return pi < pj
		}
		return s.lastUseOf(candidates[i]).Before(s.lastUseOf(candidates[j]))
	})

	// VRAM isn't attributed per model, so each eviction is assumed to free
	// an equal share of the total currently used.
	shareMB := usedMB / int64(len(running))

	var evict []string
	for _, m := range candidates {
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
