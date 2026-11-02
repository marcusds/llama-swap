package perf

import (
	"bufio"
	"context"
	"os/exec"
	"strings"
	"time"

	"github.com/mostlygeek/llama-swap/internal/logmon"
)

const nvidiaSmiQuery = "--query-gpu=index,name,uuid,temperature.gpu,utilization.gpu,memory.used,memory.total,fan.speed,power.draw"

// tryNvidiaSmi polls nvidia-smi once per interval and sends one snapshot per
// poll holding every GPU. Returns ErrNoGpuTool if nvidia-smi is not available.
//
// It runs nvidia-smi per tick instead of using --loop, which prints one line
// per GPU with nothing marking where a sample ends: forwarding each line as
// its own snapshot made consumers see only one GPU at a time.
func tryNvidiaSmi(ctx context.Context, every time.Duration, logger *logmon.Monitor) (chan []GpuStat, error) {
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		return nil, ErrNoGpuTool
	}
	if every < time.Second {
		every = time.Second
	}
	const pollTimeout = 5 * time.Second

	ch := make(chan []GpuStat, 1)
	poll := func() {
		pollCtx, cancel := context.WithTimeout(ctx, pollTimeout)
		defer cancel()
		out, err := exec.CommandContext(pollCtx, "nvidia-smi", nvidiaSmiQuery, "--format=csv,noheader,nounits").Output()
		if err != nil {
			if ctx.Err() == nil {
				logger.Debugf("nvidia-smi poll failed: %s", err.Error())
			}
			return
		}
		if stats := ParseNvidiaSmiOutput(string(out)); len(stats) > 0 {
			select {
			case ch <- stats:
			default:
			}
		}
	}

	go func() {
		defer close(ch)
		poll()
		ticker := time.NewTicker(every)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				poll()
			}
		}
	}()

	return ch, nil
}

// ParseNvidiaSmiOutput parses one nvidia-smi query: one CSV line per GPU.
// Unparseable lines are skipped.
func ParseNvidiaSmiOutput(out string) []GpuStat {
	var stats []GpuStat
	scanner := bufio.NewScanner(strings.NewReader(out))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if stat := ParseNvidiaSmiLine(line); stat != nil {
			stats = append(stats, *stat)
		}
	}
	return stats
}
