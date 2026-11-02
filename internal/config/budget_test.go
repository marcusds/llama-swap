package config

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateBudget_Basic(t *testing.T) {
	models := makeModels("gemma", "qwen", "mistral")

	mb := BudgetConfig{
		LimitMB: 16000,
		Models: map[string]BudgetModel{
			"gemma": {Priority: 10},
			"qwen":  {Priority: 5},
		},
	}

	require.NoError(t, ValidateBudget(mb, models))
}

func TestValidateBudget_MissingLimit(t *testing.T) {
	models := makeModels("gemma")
	mb := BudgetConfig{
		Models: map[string]BudgetModel{"gemma": {}},
	}

	err := ValidateBudget(mb, models)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "limit")
}

func TestValidateBudget_UnknownModel(t *testing.T) {
	models := makeModels("gemma")
	mb := BudgetConfig{
		LimitMB: 1000,
		Models:  map[string]BudgetModel{"unknown": {}},
	}

	err := ValidateBudget(mb, models)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown model")
}

func TestLoadConfig_BudgetValid(t *testing.T) {
	yamlStr := `
models:
  gemma:
    cmd: echo gemma
    proxy: http://localhost:8999
  qwen:
    cmd: echo qwen
    proxy: http://localhost:8998
routing:
  router:
    use: budget
    settings:
      budget:
        limitMB: 16384
        models:
          gemma:
            priority: 10
          qwen:
            priority: 1
`
	cfg, err := LoadConfigFromReader(strings.NewReader(yamlStr))
	require.NoError(t, err)
	require.NotNil(t, cfg.Routing.Router.Settings.Budget)
	assert.Equal(t, int64(16*1024), cfg.Routing.Router.Settings.Budget.LimitMB)
	assert.Equal(t, 10, cfg.Routing.Router.Settings.Budget.Models["gemma"].Priority)
	assert.Equal(t, 1, cfg.Routing.Router.Settings.Budget.Models["qwen"].Priority)
}

func TestMergeModelPriorities_FromModelConfig(t *testing.T) {
	p10, p3 := 10, 3
	models := makeModels("gemma", "qwen", "mistral")
	models["gemma"] = ModelConfig{Cmd: "echo gemma", Priority: &p10}
	models["qwen"] = ModelConfig{Cmd: "echo qwen", Priority: &p3}

	mb := BudgetConfig{LimitMB: 16000}
	MergeModelPriorities(&mb, models)

	assert.Equal(t, 10, mb.Models["gemma"].Priority)
	assert.Equal(t, 3, mb.Models["qwen"].Priority)

	// models without a priority are not added, and default to 0 at lookup
	_, found := mb.Models["mistral"]
	assert.False(t, found)

	require.NoError(t, ValidateBudget(mb, models))
}

func TestMergeModelPriorities_BudgetBlockWins(t *testing.T) {
	p1 := 1
	models := makeModels("gemma")
	models["gemma"] = ModelConfig{Cmd: "echo gemma", Priority: &p1}

	mb := BudgetConfig{
		LimitMB: 16000,
		Models:  map[string]BudgetModel{"gemma": {Priority: 99}},
	}
	MergeModelPriorities(&mb, models)

	assert.Equal(t, 99, mb.Models["gemma"].Priority)
}

func TestMergeModelPriorities_ZeroIsExplicit(t *testing.T) {
	zero := 0
	models := makeModels("gemma")
	models["gemma"] = ModelConfig{Cmd: "echo gemma", Priority: &zero}

	mb := BudgetConfig{LimitMB: 16000}
	MergeModelPriorities(&mb, models)

	entry, found := mb.Models["gemma"]
	require.True(t, found)
	assert.Equal(t, 0, entry.Priority)
}

func TestBudget_ModelLevelPriorityFromYAML(t *testing.T) {
	cfg, err := LoadConfigFromReader(strings.NewReader(`
routing:
  router:
    use: budget
    settings:
      budget:
        limitMB: 92160
        models:
          mistral:
            priority: 7

models:
  gemma:
    cmd: echo gemma ${PORT}
    priority: 10
  qwen:
    cmd: echo qwen ${PORT}
    priority: 5
  mistral:
    cmd: echo mistral ${PORT}
    priority: 1
  llama:
    cmd: echo llama ${PORT}
`))
	require.NoError(t, err)

	mb := cfg.Routing.Router.Settings.Budget
	require.NotNil(t, mb)
	assert.Equal(t, int64(90*1024), mb.LimitMB)
	assert.Equal(t, 10, mb.Models["gemma"].Priority)
	assert.Equal(t, 5, mb.Models["qwen"].Priority)
	// budget.models entry wins over the model's own priority
	assert.Equal(t, 7, mb.Models["mistral"].Priority)

	_, found := mb.Models["llama"]
	assert.False(t, found)
}

// An inactive budget block is accepted like inactive capacity/matrix
// settings, but reported as ignored.
func TestBudget_InactiveSettingsWarn(t *testing.T) {
	cfg, err := LoadConfigFromReader(strings.NewReader(`
models:
  gemma:
    cmd: echo gemma ${PORT}
routing:
  router:
    use: group
    settings:
      budget:
        limitMB: 1000
`))
	require.NoError(t, err)
	warnings := BudgetWarnings(cfg)
	require.Len(t, warnings, 1)
	assert.Contains(t, warnings[0], "routing.router.settings.budget is ignored")
}

func TestBudget_ModelPriorityWithoutBudgetRouterWarns(t *testing.T) {
	cfg, err := LoadConfigFromReader(strings.NewReader(`
models:
  gemma:
    cmd: echo gemma ${PORT}
    priority: 10
  qwen:
    cmd: echo qwen ${PORT}
`))
	require.NoError(t, err)
	warnings := BudgetWarnings(cfg)
	require.Len(t, warnings, 1)
	assert.Contains(t, warnings[0], "[gemma]")
	assert.Contains(t, warnings[0], "fifo.priority")
}

func TestBudget_NoWarningsWhenBudgetActive(t *testing.T) {
	cfg, err := LoadConfigFromReader(strings.NewReader(`
models:
  gemma:
    cmd: echo gemma ${PORT}
    priority: 10
routing:
  router:
    use: budget
    settings:
      budget:
        limitMB: 1000
`))
	require.NoError(t, err)
	assert.Empty(t, BudgetWarnings(cfg))
}
