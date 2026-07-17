package router

import (
	"fmt"
	"slices"
	"sort"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/process"
	gopsprocess "github.com/shirou/gopsutil/v4/process"
)

type MemoryBudget struct {
	*baseRouter
}

func NewMemoryBudget(conf config.Config, proxylog, upstreamlog *logmon.Monitor) (*MemoryBudget, error) {
	mb := conf.Routing.Router.Settings.MemoryBudget
	if mb == nil {
		return nil, fmt.Errorf("memoryBudget router requires a memoryBudget configuration")
	}

	processes := make(map[string]process.Process, len(conf.Models))
	swapper := &memoryBudgetSwapper{
		limitMB:   mb.Limit.MB(),
		priority:  make(map[string]int, len(mb.Models)),
		memoryMB:  make(map[string]int64),
		processes: processes,
		logger:    proxylog,
		sampleRSS: processRSSMB,
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

// memoryBudgetSwapper decides evictions by keeping the sum of running models'
// live-measured memory (RSS) under a configured limit, evicting
// lowest-priority (ties broken by least-recently-used) models first, and
// continuing through higher-priority models too if that's what it takes to
// fit the requested model.
//
// Memory usage is not declared in config: it's sampled live from each
// process's RSS at the start of every eviction decision, so the estimate for
// a model that has never run is 0 (optimistic) until it has been observed
// running at least once.
//
// memoryMB is mutated by EvictionFor. Like every Swapper, this is only ever
// called from the router's single run-loop goroutine, so no locking is
// needed for that state.
type memoryBudgetSwapper struct {
	limitMB   int64
	priority  map[string]int   // model ID -> priority, default 0
	memoryMB  map[string]int64 // model ID -> last-measured RSS in MB, default 0 until measured
	processes map[string]process.Process
	logger    *logmon.Monitor

	// sampleRSS reads a PID's resident set size in MB. Defaults to
	// processRSSMB; tests override it to avoid depending on real OS processes.
	sampleRSS func(pid int) (int64, error)
}

func (s *memoryBudgetSwapper) EvictionFor(target string, running []string) []string {
	s.sampleAll(running)

	if slices.Contains(running, target) {
		return nil
	}

	needed := s.memoryMB[target]
	var used int64
	for _, m := range running {
		used += s.memoryMB[m]
	}
	if used+needed <= s.limitMB {
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

	var evict []string
	for _, m := range candidates {
		if used+needed <= s.limitMB {
			break
		}
		evict = append(evict, m)
		used -= s.memoryMB[m]
	}
	return evict
}

func (s *memoryBudgetSwapper) OnSwapStart(target string, running []string) {
	evict := s.EvictionFor(target, running)
	switch {
	case len(evict) > 0:
		s.logger.Infof("memoryBudget: model=%s needMB=%d usedMB=%d limitMB=%d evict=%v",
			target, s.memoryMB[target], s.usedMB(running), s.limitMB, evict)
	case len(running) == 0:
		s.logger.Infof("memoryBudget: model=%s starting (no models running)", target)
	default:
		s.logger.Debugf("memoryBudget: model=%s fits without eviction", target)
	}
}

func (s *memoryBudgetSwapper) usedMB(running []string) int64 {
	var used int64
	for _, m := range running {
		used += s.memoryMB[m]
	}
	return used
}

func (s *memoryBudgetSwapper) lastUseOf(modelID string) time.Time {
	if p, ok := s.processes[modelID]; ok {
		return p.LastUse()
	}
	return time.Time{}
}

// sampleAll refreshes the live RSS estimate for every currently running
// model so eviction decisions use up-to-date numbers.
func (s *memoryBudgetSwapper) sampleAll(running []string) {
	for _, modelID := range running {
		p, ok := s.processes[modelID]
		if !ok {
			continue
		}
		pid, ok := p.Pid()
		if !ok {
			continue
		}
		rssMB, err := s.sampleRSS(pid)
		if err != nil {
			s.logger.Debugf("memoryBudget: failed to sample RSS for model=%s pid=%d: %v", modelID, pid, err)
			continue
		}
		s.memoryMB[modelID] = rssMB
	}
}

// processRSSMB returns pid's resident set size in megabytes.
func processRSSMB(pid int) (int64, error) {
	proc, err := gopsprocess.NewProcess(int32(pid))
	if err != nil {
		return 0, err
	}
	info, err := proc.MemoryInfo()
	if err != nil {
		return 0, err
	}
	return int64(info.RSS / (1024 * 1024)), nil
}
