//go:build linux

package process

import (
	"os"
	"strconv"
	"strings"
)

// Descendants returns pid and all of its descendant PIDs, read from /proc.
// A model launched through a wrapper (sh -c, a launcher script) does its GPU
// work in a child, so per-process GPU usage has to be matched against the
// whole tree. Processes in another PID namespace's tree (for example a
// container started by a daemon rather than as a child) are not found.
func Descendants(pid int) map[int]bool {
	return DescendantsFrom(pid, readPPIDs("/proc"))
}

// readPPIDs maps every PID under procDir to its parent PID.
func readPPIDs(procDir string) map[int]int {
	entries, err := os.ReadDir(procDir)
	if err != nil {
		return nil
	}
	ppid := make(map[int]int, len(entries))
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		data, err := os.ReadFile(procDir + "/" + e.Name() + "/stat")
		if err != nil {
			continue
		}
		if parent, ok := parseStatPPID(string(data)); ok {
			ppid[pid] = parent
		}
	}
	return ppid
}

// parseStatPPID extracts the parent PID from a /proc/<pid>/stat line. The
// command name is in parentheses and may itself contain spaces or
// parentheses, so fields are counted from the last ')'.
func parseStatPPID(stat string) (int, bool) {
	i := strings.LastIndexByte(stat, ')')
	if i < 0 {
		return 0, false
	}
	fields := strings.Fields(stat[i+1:])
	// fields[0] is the state, fields[1] the parent PID.
	if len(fields) < 2 {
		return 0, false
	}
	parent, err := strconv.Atoi(fields[1])
	if err != nil {
		return 0, false
	}
	return parent, true
}
