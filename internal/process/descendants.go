package process

// DescendantsFrom returns root and every PID whose parent chain in ppid
// (PID -> parent PID) leads to root.
func DescendantsFrom(root int, ppid map[int]int) map[int]bool {
	children := make(map[int][]int, len(ppid))
	for pid, parent := range ppid {
		if pid != parent {
			children[parent] = append(children[parent], pid)
		}
	}
	out := map[int]bool{root: true}
	stack := []int{root}
	for len(stack) > 0 {
		pid := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for _, c := range children[pid] {
			if !out[c] {
				out[c] = true
				stack = append(stack, c)
			}
		}
	}
	return out
}
