package perf

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// procVRAMFunc queries per-process VRAM use, returning PID -> MB.
type procVRAMFunc func(ctx context.Context) (map[int]int64, error)

// procVRAMTimeout bounds one per-process VRAM query.
const procVRAMTimeout = 5 * time.Second

// nvidiaProcVRAM returns a procVRAMFunc backed by nvidia-smi, or nil when
// nvidia-smi is not on PATH. Other vendors have no per-process query here.
func nvidiaProcVRAM() procVRAMFunc {
	path, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return nil
	}
	return func(ctx context.Context) (map[int]int64, error) {
		ctx, cancel := context.WithTimeout(ctx, procVRAMTimeout)
		defer cancel()
		out, err := exec.CommandContext(ctx, path,
			"--query-compute-apps=pid,used_memory",
			"--format=csv,noheader,nounits",
		).Output()
		if err != nil {
			return nil, fmt.Errorf("nvidia-smi --query-compute-apps: %w", err)
		}
		return ParseNvidiaComputeApps(string(out)), nil
	}
}

// ParseNvidiaComputeApps parses `nvidia-smi
// --query-compute-apps=pid,used_memory --format=csv,noheader,nounits` output
// into PID -> MB. A PID using several GPUs appears once per GPU and is summed.
// Lines whose memory is not a number (for example "[N/A]" on hosts that do
// not report per-process memory) are skipped.
func ParseNvidiaComputeApps(out string) map[int]int64 {
	res := make(map[int]int64)
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Split(strings.TrimSpace(line), ",")
		if len(fields) < 2 {
			continue
		}
		pid, err := strconv.Atoi(strings.TrimSpace(fields[0]))
		if err != nil || pid <= 0 {
			continue
		}
		mb, err := strconv.ParseInt(strings.TrimSpace(fields[1]), 10, 64)
		if err != nil || mb < 0 {
			continue
		}
		res[pid] += mb
	}
	return res
}

// pollProcVRAM queries per-process VRAM every interval, but only once
// ProcessVRAMMB has been called (so hosts not using the budget router never
// run the extra query).
func (m *Monitor) pollProcVRAM(ctx context.Context, every time.Duration, query procVRAMFunc) {
	tick := time.NewTicker(every)
	defer tick.Stop()
	warned := false
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		if !m.procWanted.Load() {
			continue
		}
		usage, err := query(ctx)
		at := time.Now()
		if err != nil {
			if !warned && ctx.Err() == nil {
				warned = true
				m.log.Debugf("per-process VRAM unavailable: %s", err.Error())
			}
			m.mutex.Lock()
			m.procVRAM, m.procVRAMAt = nil, time.Time{}
			m.mutex.Unlock()
			continue
		}
		m.mutex.Lock()
		m.procVRAM, m.procVRAMAt = usage, at
		m.mutex.Unlock()
	}
}

// ProcessVRAMMB returns the latest per-process VRAM use (PID -> MB) and when
// it was sampled. ok is false when no per-process data is available (no
// nvidia-smi, query failing, or no sample yet). The first call starts the
// per-process query; it is not run until something asks for it.
func (m *Monitor) ProcessVRAMMB() (map[int]int64, time.Time, bool) {
	m.procWanted.Store(true)
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	if m.procVRAM == nil {
		return nil, time.Time{}, false
	}
	cp := make(map[int]int64, len(m.procVRAM))
	for k, v := range m.procVRAM {
		cp[k] = v
	}
	return cp, m.procVRAMAt, true
}
