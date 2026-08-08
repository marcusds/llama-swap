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
// usage independent of any swap, when nothing is actively loading. See that
// method's doc comment for why this exists alongside the swap-start check in
// EvictionFor.
//
// memoryBudgetActiveLoadCheckInterval is the tighter interval used instead
// while at least one model is StateStarting. A load can push usage from under
// budget to well over it in well under 15s, and the fixed interval leaves
// that entire window unmonitored between ticks — which is exactly the shape
// of the incident that motivated this: usage climbed to under 1GB free while
// a load was in flight, with the periodic check having no reason to look
// again for several more seconds. Checking more often only while a load is
// actually in progress also gives observeAndLearn a tighter, cleaner
// before/after window per tick — less time for unrelated host activity to
// leak into a delta it's trying to attribute to a model (see the learned
// field's doc comment on memoryBudgetSwapper for why that matters on the
// unified-memory fallback).
const (
	memoryBudgetCheckInterval           = 15 * time.Second
	memoryBudgetActiveLoadCheckInterval = 2 * time.Second
)

type MemoryBudget struct {
	*baseRouter
	swapper *memoryBudgetSwapper

	// lastEvictUsedMB is the usage sampled at the tick where
	// enforceBudgetOnce last evicted, or 0 when the last tick evicted
	// nothing. See enforceBudgetOnce for why a later tick that measures no
	// drop must not evict again. Only touched by the single
	// enforceBudgetPeriodically goroutine (and by tests, serially), so it
	// needs no synchronization.
	lastEvictUsedMB int64

	// prevReady and prevUsedMB are the ready-model set and usage sampled at
	// the previous tick, used by observeAndLearn to attribute a usage delta
	// to whatever models finished loading in between. prevOK is false until
	// the first tick has run. Only touched by the single
	// enforceBudgetPeriodically goroutine, so these need no synchronization
	// (unlike swapper.learned, which observeAndLearn writes but EvictionFor
	// reads from the run-loop goroutine).
	prevReady  map[string]bool
	prevUsedMB int64
	prevOK     bool

	// lastEvictReady is the ready-model set at the tick where enforceBudgetOnce
	// last evicted, alongside lastEvictUsedMB. See enforceBudgetOnce: usage not
	// dropping since that tick only means eviction isn't helping if the ready
	// set hasn't changed either. If it has — some other model finished loading
	// in between — that alone explains a rise that has nothing to do with
	// whether the eviction worked, and pausing here would disable protection
	// at exactly the moment a concurrent load burst needs it most.
	lastEvictReady map[string]bool

	// warnedEvictFreedNothing tracks whether the pause warning already fired
	// for the current pause episode. Unlike warnLastModelOver (a structural
	// condition that only a config change resolves, so warning once ever is
	// enough), a pause episode ends whenever usage drops back under budget —
	// see the usedMB<=limitMB branch in enforceBudgetOnce, which resets this
	// alongside lastEvictUsedMB/lastEvictReady so a later, distinct pause
	// episode warns again instead of logging silently.
	warnedEvictFreedNothing bool
	warnLastModelOver       sync.Once
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
	timer := time.NewTimer(memoryBudgetCheckInterval)
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

// nextCheckInterval returns memoryBudgetActiveLoadCheckInterval if any model
// is currently StateStarting, else memoryBudgetCheckInterval. Called after
// each tick's work, so the tightened cadence takes effect for the tick
// immediately following whatever just started loading.
func (mb *MemoryBudget) nextCheckInterval() time.Duration {
	for _, st := range mb.RunningModels() {
		if st == process.StateStarting {
			return memoryBudgetActiveLoadCheckInterval
		}
	}
	return memoryBudgetCheckInterval
}

// observeAndLearn updates each model's learned footprint (see the learned
// field's doc comment on memoryBudgetSwapper) from what changed between this
// tick and the last: a rise in usedMB, attributed evenly across whatever
// models finished loading (transitioned to ready) since the previous tick.
// Concurrent loads of models with no prior observation are still invisible
// to reservedForInFlightMB — there is nothing to learn from before their
// first load — but every load after that is covered, including concurrent
// repeats of the exact burst that caught this blind spot the first time.
func (mb *MemoryBudget) observeAndLearn() {
	usedMB, src := mb.swapper.usedMemoryMB()
	if src == sourceNone {
		mb.prevOK = false
		return
	}

	states := mb.RunningModels()
	ready := make(map[string]bool, len(states))
	for id, st := range states {
		if st == process.StateReady {
			ready[id] = true
		}
	}

	if mb.prevOK {
		var newlyReady []string
		for id := range ready {
			if !mb.prevReady[id] {
				newlyReady = append(newlyReady, id)
			}
		}
		// A negative or zero delta means something was evicted in between
		// too, or usage otherwise fell — not a usable signal for what the
		// newly-ready models cost, so skip learning this tick rather than
		// record a bogus (or negative) footprint.
		if delta := usedMB - mb.prevUsedMB; len(newlyReady) > 0 && delta > 0 {
			mb.swapper.recordLearned(newlyReady, delta/int64(len(newlyReady)))
		}
	}

	mb.prevReady = ready
	mb.prevUsedMB = usedMB
	mb.prevOK = true
}

func (mb *MemoryBudget) enforceBudgetOnce() {
	usedMB, src := mb.swapper.usedMemoryMB()
	if src == sourceNone {
		return
	}
	if usedMB <= mb.swapper.limitMB {
		// Back under budget: whatever we evicted last worked, so let a
		// future overshoot evict again.
		mb.lastEvictUsedMB = 0
		mb.lastEvictReady = nil
		mb.warnedEvictFreedNothing = false
		mb.logger.Debugf("memoryBudget: periodic check usedMB=%d limitMB=%d (under budget)", usedMB, mb.swapper.limitMB)
		return
	}

	states := mb.RunningModels()
	running := make([]string, 0, len(states))
	ready := make([]string, 0, len(states))
	readySet := make(map[string]bool, len(states))
	for id, st := range states {
		running = append(running, id)
		if st == process.StateReady {
			ready = append(ready, id)
			readySet[id] = true
		}
	}

	// An earlier tick evicted and usage has not dropped since — but that only
	// means eviction isn't helping if the ready set hasn't changed either. A
	// concurrent load finishing in between raises usedMB for reasons that have
	// nothing to do with whether the eviction worked, and on a host taking a
	// burst of loads that can happen every tick — pausing here would disable
	// protection for as long as the burst continues, exactly when it's needed
	// most. Only pause when usage sat still AND nothing newly finished loading
	// to explain it: that combination is the actual signature of memory held
	// outside the models llama-swap manages (the unified-memory fallback in
	// usedMemoryMB counts everything on the host, not just models).
	if mb.lastEvictUsedMB != 0 && usedMB >= mb.lastEvictUsedMB && !hasNewMember(readySet, mb.lastEvictReady) {
		if !mb.warnedEvictFreedNothing {
			mb.warnedEvictFreedNothing = true
			mb.logger.Warnf("memoryBudget: usedMB=%d limitMB=%d still over budget, the ready set is unchanged, and the previous eviction (at %dMB) froze nothing — pausing periodic eviction until usage drops. Memory over the limit is likely held outside the models llama-swap manages.", usedMB, mb.swapper.limitMB, mb.lastEvictUsedMB)
		}
		return
	}

	if len(ready) == 0 {
		// Over budget with nothing evictable yet (e.g. the only running
		// model is still starting) — nothing to do until the next tick.
		mb.logger.Debugf("memoryBudget: periodic check usedMB=%d limitMB=%d over budget but no ready model to evict yet", usedMB, mb.swapper.limitMB)
		return
	}

	// No reservation here: this runs off a timer rather than at admission, so
	// usedMB is a plain measurement of what is actually resident.
	evict := mb.swapper.pickEvictions(ready, usedMB, 0, len(running))

	// Never evict every running model. EvictionFor gets this for free — the
	// incoming target is not in its candidate list, so something always
	// survives — but periodic enforcement has no incoming model, and a
	// single model whose own footprint exceeds the limit would otherwise be
	// killed here, reloaded by the next request, and killed again a tick
	// later, forever.
	if maxEvict := len(running) - 1; len(evict) > maxEvict {
		evict = evict[:maxEvict]
	}
	if len(evict) == 0 {
		mb.warnLastModelOver.Do(func() {
			mb.logger.Warnf("memoryBudget: usedMB=%d limitMB=%d over budget with only one model running — keeping it rather than evicting into an empty host. Raise the limit or lower the model's memory use.", usedMB, mb.swapper.limitMB)
		})
		return
	}
	mb.logger.Infof("memoryBudget: periodic check usedMB=%d limitMB=%d evicting=%v (over budget with no new model loading to trigger EvictionFor)", usedMB, mb.swapper.limitMB, evict)
	mb.lastEvictUsedMB = usedMB
	mb.lastEvictReady = readySet
	mb.Unload(0, evict...)
}

// hasNewMember reports whether current contains any key absent from previous
// — i.e. something is ready now that wasn't at the last eviction. A ready set
// that only shrank or stayed the same (the expected shape of eviction taking
// effect) returns false: only growth signals a concurrent load that could
// explain a usage rise unrelated to whether the eviction helped.
func hasNewMember(current, previous map[string]bool) bool {
	for k := range current {
		if !previous[k] {
			return true
		}
	}
	return false
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

	// learned holds each model's recent observed footprints in MB (most
	// recent last, capped at learnedHistoryLen), filled in by observeAndLearn
	// as models finish loading. EvictionFor reserves budget for models that
	// are running-but-not-ready (still loading) from the median of a model's
	// history — see learnedFootprintLocked — so a burst of concurrent loads
	// doesn't all sail through the "fits without eviction" check against a
	// usedMB sample that predates every one of them finishing.
	//
	// A single observation can be noise rather than signal: usedMemoryMB's
	// unified-memory fallback counts every process on the host, so an
	// unrelated allocation (another container, a training job, ...) that
	// happens to land in the same window as a load gets attributed to that
	// model. Trusting the latest observation outright — as if every sample
	// were clean VRAM — would let that noise drive real eviction decisions.
	// The median of several observations is far more robust: a model's real
	// footprint recurs every time it loads, so it dominates the median, while
	// unrelated noise varies and gets outvoted. A model with too few
	// observations yet (see minLearnedSamples) contributes no reservation,
	// same as one never observed at all — not enough signal yet to trust.
	learnedMu sync.RWMutex
	learned   map[string][]int64

	warnUnified sync.Once
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
func (s *memoryBudgetSwapper) recordLearned(ids []string, mb int64) {
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
func (s *memoryBudgetSwapper) learnedFootprintLocked(id string) int64 {
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

// reservedForInFlightMB sums the learned footprint of every model in running
// that is committed to load but not yet reflected in a usedMB sample — that
// is, StateStarting, or StateStopped for a swap target the scheduler has
// committed to but not started yet (running carries those via activeTargets).
//
// Ready models are already in the sample. So are stopping ones, and they are
// on their way *out* of it: running includes StateStopping (see
// baseRouter.RunningModels), so counting those would add the footprint of a
// model whose memory is still in the sample and about to be freed, on top of
// the sample that already has it — inflating usage at exactly the moment an
// eviction is in progress and cascading into evicting more than needed.
//
// Models with fewer than minLearnedSamples observations (including none at
// all) contribute 0: not enough signal yet to trust over noise, same as the
// blind spot this exists to shrink but cannot fully close.
func (s *memoryBudgetSwapper) reservedForInFlightMB(running []string) int64 {
	s.learnedMu.RLock()
	defer s.learnedMu.RUnlock()
	var reserved int64
	for _, id := range running {
		p, ok := s.processes[id]
		if !ok {
			continue
		}
		if st := p.State(); st == process.StateReady || st == process.StateStopping {
			continue
		}
		reserved += s.learnedFootprintLocked(id)
	}
	return reserved
}

// memorySource identifies which pool a usedMemoryMB sample was read from.
type memorySource int

const (
	// sourceNone means no sample was available.
	sourceNone memorySource = iota
	// sourceVRAM is total VRAM across all GPUs: models and nothing else of
	// consequence, so a rise in it is attributable to a model.
	sourceVRAM
	// sourceSysMem is the unified-memory fallback, which counts every
	// process on the host rather than only models.
	sourceSysMem
)

// usedMemoryMB samples memory used by the pool the budget applies to, and
// reports which pool that was.
//
// Normally that is total VRAM across all GPUs. Hosts with unified CPU/GPU
// memory (NVIDIA Grace-Blackwell GB10, for example) have no separate
// framebuffer, so nvidia-smi reports memory.used as 0 even while models are
// resident; budgeting against a permanent 0 would silently disable eviction.
// There the pool is system RAM, so fall back to system memory used (the same
// figure `free` reports as used) and note the switch once.
//
// The source is sourceNone when neither has produced a sample yet.
func (s *memoryBudgetSwapper) usedMemoryMB() (int64, memorySource) {
	if usedMB, ok := s.sampleVRAMUsedMB(); ok && usedMB > 0 {
		return usedMB, sourceVRAM
	}
	if s.sampleSysMemUsedMB == nil {
		return 0, sourceNone
	}
	usedMB, ok := s.sampleSysMemUsedMB()
	if !ok {
		return 0, sourceNone
	}
	s.warnUnified.Do(func() {
		s.logger.Warnf("memoryBudget: GPU monitoring reports 0MB VRAM used, budgeting against system memory used (%dMB) instead. Expected on hosts with unified CPU/GPU memory (for example NVIDIA Grace-Blackwell GB10); note that system memory used counts everything on the host, not only models.", usedMB)
	})
	return usedMB, sourceSysMem
}

func (s *memoryBudgetSwapper) EvictionFor(target string, running []string) []string {
	if slices.Contains(running, target) || len(running) == 0 {
		return nil
	}

	sampledMB, src := s.usedMemoryMB()
	if src == sourceNone {
		return nil
	}
	reservedMB := s.reservedForInFlightMB(running)
	if sampledMB+reservedMB <= s.limitMB {
		return nil
	}

	return s.pickEvictions(running, sampledMB, reservedMB, len(running))
}

// pickEvictions greedily selects the lowest-priority (ties broken by
// least-recently-used) models from candidates until sampledMB+reservedMB,
// decremented by an equal share per eviction, would drop to or below the
// limit. shareDivisor is the size of the pool the share is computed against —
// usually len(candidates), but callers evicting from a subset of the running
// set (e.g. periodic enforcement skipping still-starting processes) pass the
// full running count so the assumed per-model share stays accurate.
//
// VRAM isn't attributed per model (see the memoryBudgetSwapper doc comment),
// so each eviction is assumed to free an equal share of the total currently
// used. Only sampledMB is divided into shares: reservedMB stands for models
// that have not loaded yet (see reservedForInFlightMB), so no amount of
// evicting ready models gives it back, and folding it into the share would
// overstate what each eviction actually frees.
func (s *memoryBudgetSwapper) pickEvictions(candidates []string, sampledMB, reservedMB int64, shareDivisor int) []string {
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

	shareMB := sampledMB / int64(shareDivisor)

	usedMB := sampledMB + reservedMB
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
	usedMB, src := s.usedMemoryMB()
	reserved := s.reservedForInFlightMB(running)
	switch {
	case len(evict) > 0:
		s.logger.Infof("memoryBudget: model=%s usedMB=%d reservedMB=%d limitMB=%d evict=%v", target, usedMB, reserved, s.limitMB, evict)
	case len(running) == 0:
		s.logger.Infof("memoryBudget: model=%s starting (no models running)", target)
	case src == sourceNone:
		s.logger.Debugf("memoryBudget: model=%s starting (no memory sample available yet)", target)
	default:
		s.logger.Debugf("memoryBudget: model=%s fits without eviction (usedMB=%d reservedMB=%d limitMB=%d)", target, usedMB, reserved, s.limitMB)
	}
}

func (s *memoryBudgetSwapper) lastUseOf(modelID string) time.Time {
	if p, ok := s.processes[modelID]; ok {
		return p.LastUse()
	}
	return time.Time{}
}
