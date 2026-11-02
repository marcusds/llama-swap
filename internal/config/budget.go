package config

import "fmt"

// BudgetConfig represents the budget configuration block.
// It is a standalone alternative to groups/matrix: instead of declaring
// explicit swap rules, models are evicted lowest-priority-first whenever
// starting a new model would push total memory usage over limit.
type BudgetConfig struct {
	LimitMB int64                  `yaml:"limitMB"` // total memory cap in megabytes
	Models  map[string]BudgetModel `yaml:"models"`  // key is model ID
}

// BudgetModel declares a model's eviction priority. Higher priority
// models are evicted last. Memory usage itself is measured live from the
// running process (RSS), not declared here.
type BudgetModel struct {
	Priority int `yaml:"priority"`
}

// MergeModelPriorities folds per-model `priority:` settings into
// mb.Models so the router only has to consult one map. An explicit entry
// under budget.models wins over the model's own `priority:` field.
func MergeModelPriorities(mb *BudgetConfig, models map[string]ModelConfig) {
	for modelID, modelConfig := range models {
		if modelConfig.Priority == nil {
			continue
		}
		if _, exists := mb.Models[modelID]; exists {
			continue
		}
		if mb.Models == nil {
			mb.Models = make(map[string]BudgetModel)
		}
		mb.Models[modelID] = BudgetModel{Priority: *modelConfig.Priority}
	}
}

// ValidateBudget validates the budget config block.
func ValidateBudget(mb BudgetConfig, models map[string]ModelConfig) error {
	if mb.LimitMB <= 0 {
		return fmt.Errorf("budget.limitMB must be a positive number of megabytes, got %d", mb.LimitMB)
	}

	for modelID := range mb.Models {
		if _, exists := models[modelID]; !exists {
			return fmt.Errorf("budget.models: unknown model %q", modelID)
		}
	}

	return nil
}
