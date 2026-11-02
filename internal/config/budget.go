package config

import (
	"fmt"
	"sort"
)

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

// BudgetWarnings reports budget settings that a loaded config ignores
// because routing.router.use is not "budget". Like the other routers'
// settings, an inactive budget block is accepted (use selects which engine
// is active), and per-model priority only means something to the budget
// router; it is unrelated to the FIFO scheduler's queue priority
// (routing.scheduler.settings.fifo.priority).
func BudgetWarnings(c Config) []string {
	if c.Routing.Router.Use == "budget" {
		return nil
	}
	var warnings []string
	if c.Routing.Router.Settings.Budget != nil {
		warnings = append(warnings, fmt.Sprintf("routing.router.settings.budget is ignored because routing.router.use is %q, not \"budget\"", c.Routing.Router.Use))
	}
	var withPriority []string
	for id, m := range c.Models {
		if m.Priority != nil {
			withPriority = append(withPriority, id)
		}
	}
	if len(withPriority) > 0 {
		sort.Strings(withPriority)
		warnings = append(warnings, fmt.Sprintf("models %v set priority, which only the budget router uses (routing.router.use is %q); for request queue order use routing.scheduler.settings.fifo.priority", withPriority, c.Routing.Router.Use))
	}
	return warnings
}
