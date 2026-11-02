package process

import "testing"

func TestProcess_DescendantsFrom(t *testing.T) {
	// 1 -> 10 -> 11 -> 12, 10 -> 13, 1 -> 20 (unrelated sibling)
	ppid := map[int]int{1: 0, 10: 1, 11: 10, 12: 11, 13: 10, 20: 1}
	got := DescendantsFrom(10, ppid)
	want := map[int]bool{10: true, 11: true, 12: true, 13: true}
	if len(got) != len(want) {
		t.Fatalf("got %v want %v", got, want)
	}
	for pid := range want {
		if !got[pid] {
			t.Fatalf("missing %d in %v", pid, got)
		}
	}
}

func TestProcess_DescendantsFromUnknownRoot(t *testing.T) {
	got := DescendantsFrom(99, map[int]int{1: 0})
	if len(got) != 1 || !got[99] {
		t.Fatalf("got %v want only the root", got)
	}
}
