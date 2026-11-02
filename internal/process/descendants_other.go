//go:build !linux

package process

// Descendants returns only pid: process trees are not walked on this
// platform, so GPU usage is matched against the exact PID.
func Descendants(pid int) map[int]bool {
	return map[int]bool{pid: true}
}
