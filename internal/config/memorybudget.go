package config

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// MemoryBudgetConfig represents the memoryBudget configuration block.
// It is a standalone alternative to groups/matrix: instead of declaring
// explicit swap rules, models are evicted lowest-priority-first whenever
// starting a new model would push total memory usage over limit.
type MemoryBudgetConfig struct {
	Limit  MemorySize                   `yaml:"limit"`
	Models map[string]MemoryBudgetModel `yaml:"models"` // key is model ID
}

// MemoryBudgetModel declares a model's eviction priority. Higher priority
// models are evicted last. Memory usage itself is measured live from the
// running process (RSS), not declared here.
type MemoryBudgetModel struct {
	Priority int `yaml:"priority"`
}

// MemorySize is a byte count parsed from either a bare number (interpreted
// as megabytes, e.g. `4096`) or a human-readable string with a unit suffix
// (e.g. `"100MB"`, `"1GB"`, `"1TB"`). Internally stored in megabytes.
type MemorySize int64

var memorySizePattern = regexp.MustCompile(`^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)\s*$`)

// MB returns the size in megabytes.
func (m MemorySize) MB() int64 {
	return int64(m)
}

func (m *MemorySize) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var raw any
	if err := unmarshal(&raw); err != nil {
		return err
	}

	switch v := raw.(type) {
	case int:
		*m = MemorySize(v)
		return nil
	case string:
		mb, err := parseMemorySize(v)
		if err != nil {
			return err
		}
		*m = MemorySize(mb)
		return nil
	default:
		return fmt.Errorf("invalid memory size %v: must be a number or a string like \"100MB\", \"1GB\", \"1TB\"", raw)
	}
}

// parseMemorySize parses strings like "100MB", "1GB", "1.5TB", "512KB", "2048"
// (bare number defaults to megabytes) and returns the size in megabytes.
func parseMemorySize(s string) (int64, error) {
	match := memorySizePattern.FindStringSubmatch(s)
	if match == nil {
		return 0, fmt.Errorf("invalid memory size %q: must be a number or a string like \"100MB\", \"1GB\", \"1TB\"", s)
	}

	value, err := strconv.ParseFloat(match[1], 64)
	if err != nil {
		return 0, fmt.Errorf("invalid memory size %q: %w", s, err)
	}

	var multiplierMB float64
	switch strings.ToUpper(match[2]) {
	case "", "MB", "M":
		multiplierMB = 1
	case "B":
		multiplierMB = 1.0 / (1024 * 1024)
	case "KB", "K":
		multiplierMB = 1.0 / 1024
	case "GB", "G":
		multiplierMB = 1024
	case "TB", "T":
		multiplierMB = 1024 * 1024
	default:
		return 0, fmt.Errorf("invalid memory size %q: unknown unit %q, expected B, KB, MB, GB, or TB", s, match[2])
	}

	mb := int64(math.Round(value * multiplierMB))
	if value > 0 && mb < 1 {
		return 0, fmt.Errorf("invalid memory size %q: rounds to less than 1MB", s)
	}

	return mb, nil
}

// ValidateMemoryBudget validates the memoryBudget config block.
func ValidateMemoryBudget(mb MemoryBudgetConfig, models map[string]ModelConfig) error {
	if mb.Limit.MB() <= 0 {
		return fmt.Errorf("memoryBudget.limit must be a positive size, got %v", mb.Limit)
	}

	for modelID := range mb.Models {
		if _, exists := models[modelID]; !exists {
			return fmt.Errorf("memoryBudget.models: unknown model %q", modelID)
		}
	}

	return nil
}
