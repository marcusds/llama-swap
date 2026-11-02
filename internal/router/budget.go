package router

import (
	"context"
	"fmt"
	"slices"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/perf"
	"github.com/mostlygeek/llama-swap/internal/process"
)

// budgetCheckInterval is how often enforceBudgetPeriodically runs while no
// model is loading. budgetActiveLoadCheckInterval is the tighter interval used
// while a model is StateStarting, since a load can push usage over the limit
// quickly. The active interval is never shorter than the perf monitor's
// sampling interval: checking more often would only re-read the same sample.
const (
	budgetCheckInterval           = 15 * time.Second
	budgetActiveLoadCheckInterval = 2 * time.Second
)

type Budget struct {
	*baseRouter
	swapper *budgetSwapper

	// sampleEvery returns the perf monitor's sampling interval. nil in tests.
	sampleEvery func() time.Duration

	// The fields below are only touched by the enforceBudgetPeriodically
	// goroutine (and by tests, serially), so they need no synchronization.

	// baseline is the last usage sample taken while every running model was
	// ready, with nothing starting or stopping, and taken after the last
	// observed state change. observeAndLearn attributes the rise from it to
	// models that became ready since.
	baseline *budgetSnapshot
	// prevRunning and prevReady are the running and ready sets at the previous
	// observeAndLearn call, used to detect state changes between calls.
	prevRunning map[string]bool
	prevReady   map[string]bool
	// lastChangeAt is the latest known time the set of resident models
	// changed. A sample taken before it does not reflect the current set.
	lastChangeAt time.Time

	// lastEvict describes the last periodic eviction, or nil when usage has
	// been back under the limit since. See enforceBudgetOnce.
	lastEvict *budgetEviction

	// warnedEvictFreedNothing is set once the pause warning fired, and reset
	// when usage drops back under the limit, so each pause episode warns once.
	warnedEvictFreedNothing bool
	warnLastModelOver       sync.Once
}

// budgetSnapshot is a usage sample together with the ready set it describes.
type budgetSnapshot struct {
	sample memSample
	ready  map[string]bool
}

// budgetEviction records a periodic eviction: the sample it was decided on,
// the ready set at that time, and when the evicted processes finished
// stopping.
type budgetEviction struct {
	sample memSample
	ready  map[string]bool
	doneAt time.Time
}

func NewBudget(conf config.Config, logs *logmon.Group, perfMon *perf.Monitor) (*Budget, error) {
	proxylog := logs.ProxyLogs
	mb := conf.Routing.Router.Settings.Budget
	if mb == nil {
		return nil, fmt.Errorf("budget router requires a budget configuration")
	}
	if perfMon == nil {
		return nil, fmt.Errorf("budget router requires performance monitoring (GPU/VRAM tracking) to be enabled")
	}

	processes := make(map[string]process.Process, len(conf.Models))
	swapper := &budgetSwapper{
		limitMB:            mb.LimitMB,
		priority:           make(map[string]int, len(mb.Models)),
		processes:          processes,
		logger:             proxylog,
		sampleVRAMUsedMB:   perfMon.VRAMUsedMB,
		sampleSysMemUsedMB: perfMon.SysMemUsedMB,
		sampleProcVRAMMB:   perfMon.ProcessVRAMMB,
		descendants:        process.Descendants,
		declared:           make(map[string]int64),
	}
	for modelID, entry := range mb.Models {
		swapper.priority[modelID] = entry.PriorityOrZero()
		if entry.SizeMB > 0 {
			swapper.declared[modelID] = entry.SizeMB
		}
	}

	base, err := newBaseRouter("budget", conf, processes, proxylog, swapper)
	if err != nil {
		return nil, fmt.Errorf("creating base router: %w", err)
	}

	// Build a process for every model, same as Matrix: any model can run
	// alone even if it has no explicit priority entry (default priority 0).
	if err := base.addProcesses(conf, logs); err != nil {
		return nil, err
	}

	r := &Budget{baseRouter: base, swapper: swapper, sampleEvery: perfMon.Every}
	go base.run()
	go r.enforceBudgetPeriodically(base.procCtx)
	return r, nil
}

// enforceBudgetPeriodically re-checks usage on a timer and evicts even when no
// new model is loading.
//
// EvictionFor only runs when a swap starts, before the incoming model has
// loaded, so the model that tips usage over the limit is never weighed
// against it there. Without a periodic check the host would stay over budget
// until the next swap.
func (mb *Budget) enforceBudgetPeriodically(ctx context.Context) {
	timer := time.NewTimer(budgetCheckInterval)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
			mb.observeAndLearn()
			mb.enforceBudgetOnce()
			timer.Reset(mb.nextCheckInterval())
		}
	}
}

// nextCheckInterval returns the active-load interval (never shorter than the
// perf monitor's sampling interval) while any model is StateStarting, else
// budgetCheckInterval.
func (mb *Budget) nextCheckInterval() time.Duration {
	for _, st := range mb.RunningModels() {
		if st == process.StateStarting {
			interval := budgetActiveLoadCheckInterval
			if mb.sampleEvery != nil {
				interval = max(interval, mb.sampleEvery())
			}
			return interval
		}
	}
	return budgetCheckInterval
}

// observeAndLearn updates learned footprints (see budgetSwapper.learned) and
// readySince.
//
// Where per-process VRAM is available (see measureProcesses), a ready model's
// footprint is read directly. Otherwise a footprint is the rise in usage from
// the baseline (a sample taken while
// the model was not yet running) to the first sample taken after it became
// ready. The weights land in memory while the model is still starting, so
// the baseline must predate the load, not just the previous tick. When
// several models finish loading against the same baseline, the rise is split
// evenly between them. Nothing is learned when a model stopped in the
// meantime (its freed memory would hide part of the rise), or when the
// sample came from a different memory source than the baseline (except a
// baseline taken before the host was classified, which carries a system
// memory reading for this case; see usedMemoryMB).
func (mb *Budget) observeAndLearn() {
	now := time.Now()
	smp := mb.swapper.usedMemoryMB()
	statuses := mb.RunningStatus()

	running := make(map[string]bool, len(statuses))
	ready := make(map[string]bool, len(statuses))
	quiet := true
	for id, st := range statuses {
		running[id] = true
		if st.State == process.StateReady {
			ready[id] = true
		} else {
			quiet = false
		}
	}

	var newlyReady []string
	for id := range ready {
		if mb.prevReady[id] {
			continue
		}
		newlyReady = append(newlyReady, id)
		at := statuses[id].ReadySince
		if at.IsZero() {
			at = now
		}
		mb.swapper.recordReadySince([]string{id}, at)
		if at.After(mb.lastChangeAt) {
			mb.lastChangeAt = at
		}
	}
	for id := range mb.prevRunning {
		if !running[id] {
			// Stopped since the last call, at an unknown time up to now.
			mb.lastChangeAt = now
		}
	}
	if !quiet {
		mb.lastChangeAt = now
	}
	mb.prevRunning = running
	mb.prevReady = ready

	// Per-process figures, where available, are clean per-model data and need
	// no quiet window.
	perProc := mb.swapper.measureProcesses(ready)

	if !quiet || smp.src == sourceNone || smp.at.Before(mb.lastChangeAt) {
		return
	}

	if b := mb.baseline; b != nil {
		baseMB, ok := b.sample.comparableMB(smp.src)
		if !ok {
			mb.baseline = &budgetSnapshot{sample: smp, ready: ready}
			return
		}
		var added []string
		removed := false
		for id := range ready {
			if !b.ready[id] {
				added = append(added, id)
			}
		}
		for id := range b.ready {
			if !ready[id] {
				removed = true
			}
		}
		// Models measured per process already have a clean figure; take
		// their share out of the rise and split the rest among the others.
		delta := smp.mb - baseMB
		var unmeasured []string
		for _, id := range added {
			if v, ok := perProc[id]; ok {
				delta -= v
			} else {
				unmeasured = append(unmeasured, id)
			}
		}
		if len(unmeasured) > 0 && !removed && delta > 0 {
			mb.swapper.recordLearned(unmeasured, delta/int64(len(unmeasured)))
		}
	}
	mb.baseline = &budgetSnapshot{sample: smp, ready: ready}
}

// enforceBudgetOnce evicts ready, idle models (lowest priority first, see
// pickEvictions) while usage is over the limit.
//
// After an eviction it waits for a sample taken after the evicted processes
// stopped. If that sample is no lower and nothing new became ready, the
// memory over the limit is not held by models llama-swap can evict, so it
// pauses (with a warning) until usage drops under the limit, instead of
// evicting every model one tick at a time.
func (mb *Budget) enforceBudgetOnce() {
	smp := mb.swapper.usedMemoryMB()
	if smp.src == sourceNone {
		return
	}
	limitMB := mb.swapper.limitMB

	if le := mb.lastEvict; le != nil {
		if le.sample.src != smp.src {
			// Samples from different pools cannot be compared.
			mb.lastEvict = nil
			mb.warnedEvictFreedNothing = false
		} else if !smp.at.After(le.doneAt) {
			mb.logger.Debugf("budget: periodic check waiting for a memory sample taken after the last eviction")
			return
		}
	}

	if smp.mb <= limitMB {
		mb.lastEvict = nil
		mb.warnedEvictFreedNothing = false
		mb.logger.Debugf("budget: periodic check usedMB=%d limitMB=%d (under budget)", smp.mb, limitMB)
		return
	}

	states := mb.RunningModels()
	ready := make([]string, 0, len(states))
	readySet := make(map[string]bool, len(states))
	for id, st := range states {
		if st == process.StateReady {
			ready = append(ready, id)
			readySet[id] = true
		}
	}

	// Usage did not drop since the last eviction, and no newly ready model
	// explains it.
	if le := mb.lastEvict; le != nil && smp.mb >= le.sample.mb && !hasNewMember(readySet, le.ready) {
		if !mb.warnedEvictFreedNothing {
			mb.warnedEvictFreedNothing = true
			mb.logger.Warnf("budget: usedMB=%d limitMB=%d still over budget, the ready set is unchanged, and the previous eviction (at %dMB) freed nothing — pausing periodic eviction until usage drops. Memory over the limit is likely held outside the models llama-swap manages.", smp.mb, limitMB, le.sample.mb)
		}
		return
	}

	if len(ready) == 0 {
		mb.logger.Debugf("budget: periodic check usedMB=%d limitMB=%d over budget but no model is ready yet", smp.mb, limitMB)
		return
	}

	// Never evict every running model: with no incoming model to make room
	// for, a single model over the limit would be evicted, reloaded by the
	// next request and evicted again.
	if len(states) <= 1 {
		mb.warnLastModelOver.Do(func() {
			mb.logger.Warnf("budget: usedMB=%d limitMB=%d over budget with only one model running — keeping it rather than evicting into an empty host. Raise the limit or lower the model's memory use.", smp.mb, limitMB)
		})
		return
	}
	maxEvict := len(states) - 1
	pool := make([]string, 0, len(states))
	for id := range states {
		pool = append(pool, id)
	}

	// The idle check and the stop are one step on the run loop, so a model
	// cannot be granted a request between them. The usage sample is a plain
	// measurement here, so nothing is reserved for loading models.
	evicted, done := mb.EvictIdle(ready, func(idle []string) []string {
		evict := mb.swapper.pickEvictions(idle, pool, smp.mb, 0)
		if len(evict) > maxEvict {
			evict = evict[:maxEvict]
		}
		return evict
	})
	if len(evicted) == 0 {
		mb.logger.Debugf("budget: periodic check usedMB=%d limitMB=%d over budget but every ready model is busy", smp.mb, limitMB)
		return
	}
	mb.logger.Infof("budget: periodic check usedMB=%d limitMB=%d evicting=%v (over budget with no new model loading to trigger EvictionFor)", smp.mb, limitMB, evicted)
	<-done
	mb.lastEvict = &budgetEviction{sample: smp, ready: readySet, doneAt: time.Now()}
}

// hasNewMember reports whether current contains any key absent from previous.
// A ready set that only shrank (the expected effect of eviction) returns
// false; growth means a load finished, which can explain a usage rise.
func hasNewMember(current, previous map[string]bool) bool {
	for k := range current {
		if !previous[k] {
			return true
		}
	}
	return false
}

// budgetSwapper decides evictions by keeping total VRAM used under a
// configured limit, evicting lowest-priority (ties broken by
// least-recently-used) models first, and continuing through higher-priority
// models too if that's what it takes to fit the requested model.
//
// The limit applies to VRAM summed across all GPUs. On a multi-GPU host one
// card can run out of memory while the total is still under the limit.
//
// On hosts with unified CPU/GPU memory, where GPU tools report 0MB VRAM used,
// the budget is applied to system memory used instead (see usedMemoryMB).
//
// Admission and eviction use a per-model size estimate (see estimateMB): a
// declared sizeMB, a per-process VRAM measurement, or a learned median. The
// measured total is the safety net: it is what must stay under the limit, and
// models with no estimate are assumed to free an equal share of whatever the
// known estimates do not account for.
type budgetSwapper struct {
	limitMB   int64
	priority  map[string]int // model ID -> priority, default 0
	processes map[string]process.Process
	logger    *logmon.Monitor

	// sampleVRAMUsedMB returns total VRAM used in MB across all GPUs and when
	// it was sampled, and false if no sample is available yet. Defaults to
	// perfMon.VRAMUsedMB; tests override it.
	sampleVRAMUsedMB func() (int64, time.Time, bool)

	// sampleSysMemUsedMB returns system memory used in MB and when it was
	// sampled, and false if no sample is available yet. Used as the budget
	// signal on unified-memory hosts. Defaults to perfMon.SysMemUsedMB.
	sampleSysMemUsedMB func() (int64, time.Time, bool)

	// sampleProcVRAMMB returns per-process VRAM use (PID -> MB) and when it
	// was sampled, and false when unavailable. Defaults to
	// perfMon.ProcessVRAMMB (NVIDIA only); nil disables per-process data.
	sampleProcVRAMMB func() (map[int]int64, time.Time, bool)

	// descendants returns a PID and all its descendants, so a model launched
	// through a wrapper is matched to the child doing the GPU work. Defaults
	// to process.Descendants.
	descendants func(pid int) map[int]bool

	// declared is each model's configured sizeMB (budget.models.<id>.sizeMB).
	// Read-only after construction.
	declared map[string]int64

	// measured is each model's latest per-process VRAM figure, kept after
	// the model stops so it still estimates the next load. procLearnedAt is
	// the readySince of the load last recorded into learned from per-process
	// data, so each load is recorded once.
	measuredMu    sync.RWMutex
	measured      map[string]int64
	procLearnedAt map[string]time.Time

	// memMode is the memoryMode usedMemoryMB has settled on.
	memMode atomic.Int32

	// learned holds each model's recent observed footprints in MB (most
	// recent last, capped at learnedHistoryLen), filled in by observeAndLearn
	// from per-process figures or, without those, usage deltas. Its median is
	// the last-resort estimate (see estimateMB).
	//
	// The median is used because a delta observation can be noise: on the
	// system-memory fallback any unrelated allocation during a load is
	// attributed to that model. A model with fewer than minLearnedSamples
	// observations has no learned estimate.
	learnedMu sync.RWMutex
	learned   map[string][]int64

	// readySince is when each model last became ready, filled in by
	// observeAndLearn. pickEvictions tries models inside
	// budgetEvictionGracePeriod of their load last, so a low-priority model is
	// not evicted the moment it finishes loading for the request waiting on
	// it.
	readySinceMu sync.RWMutex
	readySince   map[string]time.Time

	// lastDecision is the sample and reservation behind the last EvictionFor
	// call, for OnSwapStart to log. Only touched on the run loop.
	lastDecision budgetDecision

	warnUnified sync.Once
}

type budgetDecision struct {
	target   string
	sample   memSample
	reserved int64
	targetMB int64 // the target's estimate, 0 when unknown
}

// budgetEvictionGracePeriod is how long a model is tried last for periodic
// eviction after becoming ready. Chosen to comfortably outlast a typical
// inference response rather than to match any particular check interval.
const budgetEvictionGracePeriod = 30 * time.Second

// recordReadySince sets readySince[id] = at for each of ids.
func (s *budgetSwapper) recordReadySince(ids []string, at time.Time) {
	s.readySinceMu.Lock()
	defer s.readySinceMu.Unlock()
	if s.readySince == nil {
		s.readySince = make(map[string]time.Time, len(ids))
	}
	for _, id := range ids {
		s.readySince[id] = at
	}
}

// recentlyReady reports whether id became ready within the last
// budgetEvictionGracePeriod. A model with no recorded readySince is not
// protected.
func (s *budgetSwapper) recentlyReady(id string) bool {
	s.readySinceMu.RLock()
	defer s.readySinceMu.RUnlock()
	t, ok := s.readySince[id]
	if !ok {
		return false
	}
	return time.Since(t) < budgetEvictionGracePeriod
}

// learnedHistoryLen is how many recent observations recordLearned keeps per
// model. minLearnedSamples is the minimum before learnedFootprintLocked
// trusts the median rather than treating the model as unobserved.
const (
	learnedHistoryLen = 5
	minLearnedSamples = 3
)

// recordLearned appends mb to each of ids' observation history, capped at the
// most recent learnedHistoryLen entries.
func (s *budgetSwapper) recordLearned(ids []string, mb int64) {
	s.learnedMu.Lock()
	defer s.learnedMu.Unlock()
	if s.learned == nil {
		s.learned = make(map[string][]int64, len(ids))
	}
	for _, id := range ids {
		hist := append(s.learned[id], mb)
		if len(hist) > learnedHistoryLen {
			hist = hist[len(hist)-learnedHistoryLen:]
		}
		s.learned[id] = hist
	}
}

// learnedFootprintLocked returns the median of id's observation history, or 0
// if there are fewer than minLearnedSamples. Callers must hold learnedMu.
func (s *budgetSwapper) learnedFootprintLocked(id string) int64 {
	hist := s.learned[id]
	if len(hist) < minLearnedSamples {
		return 0
	}
	sorted := append([]int64(nil), hist...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	n := len(sorted)
	if n%2 == 1 {
		return sorted[n/2]
	}
	return (sorted[n/2-1] + sorted[n/2]) / 2
}

// reservedForInFlightMB sums the estimate (see estimateMB) of every model in running
// that is committed to load but not yet reflected in a usage sample:
// StateStarting, or StateStopped for a swap target the scheduler has
// committed to but not started yet.
//
// Ready models are already in the sample. Stopping models are too, and
// their memory is about to be freed, so reserving for them would count them
// twice and evict more than needed.
func (s *budgetSwapper) reservedForInFlightMB(running []string) int64 {
	var reserved int64
	for _, id := range running {
		p, ok := s.processes[id]
		if !ok {
			continue
		}
		if st := p.State(); st == process.StateReady || st == process.StateStopping {
			continue
		}
		if est, ok := s.estimateMB(id); ok {
			reserved += est
		}
	}
	return reserved
}

// estimateMB returns id's expected footprint once loaded: its declared
// sizeMB, else its latest per-process VRAM measurement, else the median of
// its learned history. ok is false when none is available.
func (s *budgetSwapper) estimateMB(id string) (int64, bool) {
	if v := s.declared[id]; v > 0 {
		return v, true
	}
	s.measuredMu.RLock()
	v, ok := s.measured[id]
	s.measuredMu.RUnlock()
	if ok && v > 0 {
		return v, true
	}
	s.learnedMu.RLock()
	v = s.learnedFootprintLocked(id)
	s.learnedMu.RUnlock()
	return v, v > 0
}

// measureProcesses reads per-process VRAM for each ready model, summed over
// the model's process tree, and updates measured. A figure counts only when
// the sample was taken after the model became ready. Each load's first such
// figure is also recorded into learned, as clean data. It returns the figures
// found this call, keyed by model ID; nil when per-process data is
// unavailable.
func (s *budgetSwapper) measureProcesses(ready map[string]bool) map[string]int64 {
	if s.sampleProcVRAMMB == nil || s.descendants == nil || len(ready) == 0 {
		return nil
	}
	usage, at, ok := s.sampleProcVRAMMB()
	if !ok || len(usage) == 0 {
		return nil
	}
	out := make(map[string]int64, len(ready))
	since := make(map[string]time.Time, len(ready))
	for id := range ready {
		p, ok := s.processes[id]
		if !ok {
			continue
		}
		pid, ok := p.Pid()
		if !ok {
			continue
		}
		var sum int64
		for d := range s.descendants(pid) {
			sum += usage[d]
		}
		if sum <= 0 {
			continue
		}
		s.readySinceMu.RLock()
		rs := s.readySince[id]
		s.readySinceMu.RUnlock()
		if at.Before(rs) {
			continue // sampled before this load finished
		}
		out[id] = sum
		since[id] = rs
	}
	if len(out) == 0 {
		return nil
	}

	var toLearn []string
	s.measuredMu.Lock()
	if s.measured == nil {
		s.measured = make(map[string]int64, len(out))
		s.procLearnedAt = make(map[string]time.Time, len(out))
	}
	for id, v := range out {
		s.measured[id] = v
		if last, seen := s.procLearnedAt[id]; !seen || !last.Equal(since[id]) {
			s.procLearnedAt[id] = since[id]
			toLearn = append(toLearn, id)
		}
	}
	s.measuredMu.Unlock()
	for _, id := range toLearn {
		s.recordLearned([]string{id}, out[id])
	}
	return out
}

// memorySource identifies which pool a usedMemoryMB sample was read from.
type memorySource int

const (
	// sourceNone means no sample was available.
	sourceNone memorySource = iota
	// sourceVRAM is total VRAM across all GPUs.
	sourceVRAM
	// sourceSysMem is the unified-memory fallback, which counts every
	// process on the host rather than only models.
	sourceSysMem
)

// memoryMode is what usedMemoryMB has concluded about the host.
type memoryMode int32

const (
	// memModeUnknown: VRAM has only read 0, with no model ready.
	memModeUnknown memoryMode = iota
	// memModeDiscrete: VRAM has read above 0, so it is tracked directly.
	memModeDiscrete
	// memModeUnified: VRAM read 0 while a model was ready.
	memModeUnified
)

// memSample is one usage reading and the pool and time it was sampled from.
type memSample struct {
	mb  int64
	src memorySource
	at  time.Time
	// sysMB is system memory used at the same time, set (hasSys) only on a
	// VRAM-0 sample taken before the host was classified discrete or unified.
	sysMB  int64
	hasSys bool
}

// comparableMB returns s's reading in the pool src, and false if s has none.
func (s memSample) comparableMB(src memorySource) (int64, bool) {
	if s.src == src {
		return s.mb, true
	}
	if src == sourceSysMem && s.hasSys {
		return s.sysMB, true
	}
	return 0, false
}

// usedMemoryMB samples memory used by the pool the budget applies to.
//
// Normally that is total VRAM across all GPUs. Hosts with unified CPU/GPU
// memory (NVIDIA Grace-Blackwell GB10, for example) report 0MB VRAM used even
// while models are resident, so there the pool is system memory used. A
// discrete GPU can also read 0 when nothing is loaded, so a 0 reading only
// switches to system memory once a model is ready, and never after VRAM has
// read above 0. With no VRAM sample at all, system memory is used.
//
// The source is sourceNone when no usable sample exists yet.
func (s *budgetSwapper) usedMemoryMB() memSample {
	vramMB, at, ok := s.sampleVRAMUsedMB()
	if ok {
		mode := memoryMode(s.memMode.Load())
		if vramMB > 0 {
			s.memMode.Store(int32(memModeDiscrete))
			return newMemSample(vramMB, sourceVRAM, at)
		}
		if mode == memModeDiscrete {
			return newMemSample(0, sourceVRAM, at)
		}
		if mode == memModeUnknown && !s.anyReady() {
			// Host type not known yet. Keep the system memory reading too, so
			// a baseline taken now still works if the host turns out unified.
			smp := newMemSample(0, sourceVRAM, at)
			if s.sampleSysMemUsedMB != nil {
				if sysMB, _, sysOK := s.sampleSysMemUsedMB(); sysOK {
					smp.sysMB, smp.hasSys = sysMB, true
				}
			}
			return smp
		}
	}
	if s.sampleSysMemUsedMB == nil {
		return memSample{}
	}
	sysMB, sysAt, sysOK := s.sampleSysMemUsedMB()
	if !sysOK {
		return memSample{}
	}
	if ok {
		s.memMode.Store(int32(memModeUnified))
		s.warnUnified.Do(func() {
			s.logger.Warnf("budget: GPU monitoring reports 0MB VRAM used while models are loaded, budgeting against system memory used (%dMB) instead. Expected on hosts with unified CPU/GPU memory (for example NVIDIA Grace-Blackwell GB10); note that system memory used counts everything on the host, not only models.", sysMB)
		})
	}
	return newMemSample(sysMB, sourceSysMem, sysAt)
}

// newMemSample builds a sample. A missing timestamp is treated as now.
func newMemSample(mb int64, src memorySource, at time.Time) memSample {
	if at.IsZero() {
		at = time.Now()
	}
	return memSample{mb: mb, src: src, at: at}
}

// anyReady reports whether any managed model is ready.
func (s *budgetSwapper) anyReady() bool {
	for _, p := range s.processes {
		if p.State() == process.StateReady {
			return true
		}
	}
	return false
}

func (s *budgetSwapper) EvictionFor(target string, running []string) []string {
	s.lastDecision = budgetDecision{target: target}
	if slices.Contains(running, target) || len(running) == 0 {
		return nil
	}

	smp := s.usedMemoryMB()
	if smp.src == sourceNone {
		return nil
	}
	reservedMB := s.reservedForInFlightMB(running)
	// With a known estimate for the target, admission makes room for it
	// before it loads; without one, only the current usage is checked.
	targetMB, _ := s.estimateMB(target)
	s.lastDecision = budgetDecision{target: target, sample: smp, reserved: reservedMB, targetMB: targetMB}
	if smp.mb+reservedMB+targetMB <= s.limitMB {
		return nil
	}

	return s.pickEvictions(running, running, smp.mb, reservedMB+targetMB)
}

// pickEvictions greedily selects the lowest-priority (ties broken by
// least-recently-used) models from candidates until sampledMB+extraMB would
// drop to or below the limit. extraMB is memory not yet in the sample that
// must also fit: reservations for loading models and the incoming target.
//
// Each evicted model is assumed to free its estimate (see estimateMB). A
// model with no estimate is assumed to free an equal share of what the known
// estimates of the resident (ready or stopping) models in pool do not account
// for, split across every model in pool with no estimate. pool is the full
// running set, which may be larger than candidates.
//
// Candidates inside their grace period (see readySince) are tried after
// every other candidate regardless of priority, but are still evicted if
// nothing else is left.
func (s *budgetSwapper) pickEvictions(candidates, pool []string, sampledMB, extraMB int64) []string {
	if len(pool) == 0 {
		return nil
	}

	var unprotected, protected []string
	for _, id := range candidates {
		if s.recentlyReady(id) {
			protected = append(protected, id)
		} else {
			unprotected = append(unprotected, id)
		}
	}
	sorted := append(s.sortByPriorityAndLRU(unprotected), s.sortByPriorityAndLRU(protected)...)

	estimates := make(map[string]int64, len(pool))
	var knownResidentMB int64
	unknown := 0
	for _, id := range pool {
		est, ok := s.estimateMB(id)
		if !ok {
			unknown++
			continue
		}
		estimates[id] = est
		if p, ok := s.processes[id]; ok {
			if st := p.State(); st == process.StateReady || st == process.StateStopping {
				knownResidentMB += est
			}
		}
	}
	var shareMB int64
	if unknown > 0 {
		shareMB = max(0, sampledMB-knownResidentMB) / int64(unknown)
	}

	usedMB := sampledMB + extraMB
	var evict []string
	for _, m := range sorted {
		if usedMB <= s.limitMB {
			break
		}
		evict = append(evict, m)
		if est, ok := estimates[m]; ok {
			usedMB -= est
		} else if est, ok := s.estimateMB(m); ok {
			usedMB -= est // candidate outside pool
		} else {
			usedMB -= shareMB
		}
	}
	return evict
}

// sortByPriorityAndLRU returns a copy of ids sorted by ascending priority
// (lowest evicted first), ties broken by least-recently-used.
func (s *budgetSwapper) sortByPriorityAndLRU(ids []string) []string {
	sorted := make([]string, len(ids))
	copy(sorted, ids)
	sort.Slice(sorted, func(i, j int) bool {
		pi, pj := s.priority[sorted[i]], s.priority[sorted[j]]
		if pi != pj {
			return pi < pj
		}
		return s.lastUseOf(sorted[i]).Before(s.lastUseOf(sorted[j]))
	})
	return sorted
}

// OnSwapStart logs the eviction the scheduler acted on, with the sample and
// reservation EvictionFor decided it from.
func (s *budgetSwapper) OnSwapStart(target string, running, evict []string) {
	d := s.lastDecision
	if d.target != target {
		d = budgetDecision{target: target}
	}
	switch {
	case len(evict) > 0:
		s.logger.Infof("budget: model=%s usedMB=%d reservedMB=%d targetMB=%d limitMB=%d evict=%v", target, d.sample.mb, d.reserved, d.targetMB, s.limitMB, evict)
	case len(running) == 0:
		s.logger.Infof("budget: model=%s starting (no models running)", target)
	case d.sample.src == sourceNone:
		s.logger.Debugf("budget: model=%s starting (no memory sample available yet)", target)
	default:
		s.logger.Debugf("budget: model=%s fits without eviction (usedMB=%d reservedMB=%d targetMB=%d limitMB=%d)", target, d.sample.mb, d.reserved, d.targetMB, s.limitMB)
	}
}

func (s *budgetSwapper) lastUseOf(modelID string) time.Time {
	if p, ok := s.processes[modelID]; ok {
		return p.LastUse()
	}
	return time.Time{}
}
