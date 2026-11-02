//go:build linux

package process

import (
	"os"
	"os/exec"
	"testing"
)

func TestProcess_ParseStatPPID(t *testing.T) {
	cases := map[string]int{
		"123 (llama-server) S 45 123 123 0 -1":     45,
		"123 (weird) name)) R 7 123 123 0 -1 4194": 7,
	}
	for line, want := range cases {
		got, ok := parseStatPPID(line)
		if !ok || got != want {
			t.Fatalf("parseStatPPID(%q) = %d,%v want %d", line, got, ok, want)
		}
	}
	if _, ok := parseStatPPID("garbage"); ok {
		t.Fatal("parsed garbage")
	}
}

func TestProcess_DescendantsFindsChild(t *testing.T) {
	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Skipf("cannot start sleep: %v", err)
	}
	defer func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	}()
	got := Descendants(os.Getpid())
	if !got[os.Getpid()] || !got[cmd.Process.Pid] {
		t.Fatalf("Descendants(%d) = %v, want self and child %d", os.Getpid(), got, cmd.Process.Pid)
	}
}
