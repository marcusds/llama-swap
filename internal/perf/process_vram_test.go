package perf

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
)

func TestParseNvidiaComputeApps(t *testing.T) {
	out := "1234, 20480\n5678, 512\n1234, 1024\n\n999, [N/A]\nbad, 12\n42\n"
	got := ParseNvidiaComputeApps(out)
	want := map[int]int64{1234: 21504, 5678: 512}
	if len(got) != len(want) {
		t.Fatalf("got %v want %v", got, want)
	}
	for pid, mb := range want {
		if got[pid] != mb {
			t.Fatalf("pid %d: got %d want %d (all: %v)", pid, got[pid], mb, got)
		}
	}
}

func TestParseNvidiaComputeApps_Empty(t *testing.T) {
	if got := ParseNvidiaComputeApps(""); len(got) != 0 {
		t.Fatalf("got %v want empty", got)
	}
}

func TestMonitor_ProcessVRAMPollsOnlyWhenWanted(t *testing.T) {
	m, err := New(config.PerformanceConfig{Every: 100 * time.Millisecond}, newTestLogger())
	if err != nil {
		t.Fatal(err)
	}
	var calls atomic.Int32
	query := func(context.Context) (map[int]int64, error) {
		calls.Add(1)
		return map[int]int64{7: 300}, nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.pollProcVRAM(ctx, 10*time.Millisecond, query)

	time.Sleep(50 * time.Millisecond)
	if n := calls.Load(); n != 0 {
		t.Fatalf("query ran %d times before anything asked for per-process data", n)
	}

	if _, _, ok := m.ProcessVRAMMB(); ok {
		t.Fatal("ok before any sample")
	}
	deadline := time.Now().Add(2 * time.Second)
	for {
		usage, at, ok := m.ProcessVRAMMB()
		if ok {
			if usage[7] != 300 || at.IsZero() {
				t.Fatalf("usage=%v at=%v", usage, at)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("no per-process sample after it was requested")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestMonitor_ProcessVRAMClearedOnQueryError(t *testing.T) {
	m, err := New(config.PerformanceConfig{Every: 100 * time.Millisecond}, newTestLogger())
	if err != nil {
		t.Fatal(err)
	}
	var fail atomic.Bool
	query := func(context.Context) (map[int]int64, error) {
		if fail.Load() {
			return nil, errors.New("boom")
		}
		return map[int]int64{7: 300}, nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m.ProcessVRAMMB()
	go m.pollProcVRAM(ctx, 10*time.Millisecond, query)

	waitFor := func(want bool) {
		t.Helper()
		deadline := time.Now().Add(2 * time.Second)
		for {
			if _, _, ok := m.ProcessVRAMMB(); ok == want {
				return
			}
			if time.Now().After(deadline) {
				t.Fatalf("ProcessVRAMMB ok never became %v", want)
			}
			time.Sleep(5 * time.Millisecond)
		}
	}
	waitFor(true)
	fail.Store(true)
	waitFor(false)
}
