package config

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateMemoryBudget_Basic(t *testing.T) {
	models := makeModels("gemma", "qwen", "mistral")

	mb := MemoryBudgetConfig{
		Limit: 16000,
		Models: map[string]MemoryBudgetModel{
			"gemma": {Priority: 10},
			"qwen":  {Priority: 5},
		},
	}

	require.NoError(t, ValidateMemoryBudget(mb, models))
}

func TestValidateMemoryBudget_MissingLimit(t *testing.T) {
	models := makeModels("gemma")
	mb := MemoryBudgetConfig{
		Models: map[string]MemoryBudgetModel{"gemma": {}},
	}

	err := ValidateMemoryBudget(mb, models)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "limit")
}

func TestValidateMemoryBudget_UnknownModel(t *testing.T) {
	models := makeModels("gemma")
	mb := MemoryBudgetConfig{
		Limit:  1000,
		Models: map[string]MemoryBudgetModel{"unknown": {}},
	}

	err := ValidateMemoryBudget(mb, models)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown model")
}

func TestLoadConfig_MemoryBudgetXORGroupsAndMatrix(t *testing.T) {
	yamlStr := `
models:
  gemma:
    cmd: echo gemma
    proxy: http://localhost:8999
groups:
  g1:
    members: ["gemma"]
memoryBudget:
  limit: 1000
`
	_, err := LoadConfigFromReader(strings.NewReader(yamlStr))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot use both 'groups' and 'memoryBudget'")

	yamlStr2 := `
models:
  gemma:
    cmd: echo gemma
    proxy: http://localhost:8999
matrix:
  vars:
    g: gemma
  sets:
    s1: "g"
memoryBudget:
  limit: 1000
`
	_, err = LoadConfigFromReader(strings.NewReader(yamlStr2))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot use both 'matrix' and 'memoryBudget'")
}

func TestLoadConfig_MemoryBudgetValid(t *testing.T) {
	yamlStr := `
models:
  gemma:
    cmd: echo gemma
    proxy: http://localhost:8999
  qwen:
    cmd: echo qwen
    proxy: http://localhost:8998
memoryBudget:
  limit: 16GB
  models:
    gemma:
      priority: 10
    qwen:
      priority: 1
`
	cfg, err := LoadConfigFromReader(strings.NewReader(yamlStr))
	require.NoError(t, err)
	require.NotNil(t, cfg.MemoryBudget)
	assert.Equal(t, int64(16*1024), cfg.MemoryBudget.Limit.MB())
	assert.Equal(t, 10, cfg.MemoryBudget.Models["gemma"].Priority)
	assert.Equal(t, 1, cfg.MemoryBudget.Models["qwen"].Priority)
	assert.Equal(t, 0, len(cfg.Groups))
}

func TestParseMemorySize(t *testing.T) {
	tests := []struct {
		in      string
		wantMB  int64
		wantErr bool
	}{
		{in: "4096", wantMB: 4096},
		{in: "100MB", wantMB: 100},
		{in: "100mb", wantMB: 100},
		{in: "1GB", wantMB: 1024},
		{in: "1TB", wantMB: 1024 * 1024},
		{in: "512KB", wantMB: 1}, // rounds up
		{in: "1.5GB", wantMB: 1536},
		{in: "0.4MB", wantErr: true},
		{in: "100XB", wantErr: true},
		{in: "not-a-size", wantErr: true},
	}

	for _, tt := range tests {
		mb, err := parseMemorySize(tt.in)
		if tt.wantErr {
			assert.Error(t, err, tt.in)
			continue
		}
		require.NoError(t, err, tt.in)
		assert.Equal(t, tt.wantMB, mb, tt.in)
	}
}

func TestMergeModelPriorities_FromModelConfig(t *testing.T) {
	p10, p3 := 10, 3
	models := makeModels("gemma", "qwen", "mistral")
	models["gemma"] = ModelConfig{Cmd: "echo gemma", Priority: &p10}
	models["qwen"] = ModelConfig{Cmd: "echo qwen", Priority: &p3}

	mb := MemoryBudgetConfig{Limit: 16000}
	MergeModelPriorities(&mb, models)

	assert.Equal(t, 10, mb.Models["gemma"].Priority)
	assert.Equal(t, 3, mb.Models["qwen"].Priority)

	// models without a priority are not added, and default to 0 at lookup
	_, found := mb.Models["mistral"]
	assert.False(t, found)

	require.NoError(t, ValidateMemoryBudget(mb, models))
}

func TestMergeModelPriorities_BudgetBlockWins(t *testing.T) {
	p1 := 1
	models := makeModels("gemma")
	models["gemma"] = ModelConfig{Cmd: "echo gemma", Priority: &p1}

	mb := MemoryBudgetConfig{
		Limit:  16000,
		Models: map[string]MemoryBudgetModel{"gemma": {Priority: 99}},
	}
	MergeModelPriorities(&mb, models)

	assert.Equal(t, 99, mb.Models["gemma"].Priority)
}

func TestMergeModelPriorities_ZeroIsExplicit(t *testing.T) {
	zero := 0
	models := makeModels("gemma")
	models["gemma"] = ModelConfig{Cmd: "echo gemma", Priority: &zero}

	mb := MemoryBudgetConfig{Limit: 16000}
	MergeModelPriorities(&mb, models)

	entry, found := mb.Models["gemma"]
	require.True(t, found)
	assert.Equal(t, 0, entry.Priority)
}

func TestMemoryBudget_ModelLevelPriorityFromYAML(t *testing.T) {
	cfg, err := LoadConfigFromReader(strings.NewReader(`
memoryBudget:
  limit: 90GB
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

	mb := cfg.Routing.Router.Settings.MemoryBudget
	require.NotNil(t, mb)
	assert.Equal(t, int64(90*1024), mb.Limit.MB())
	assert.Equal(t, 10, mb.Models["gemma"].Priority)
	assert.Equal(t, 5, mb.Models["qwen"].Priority)
	// memoryBudget.models entry wins over the model's own priority
	assert.Equal(t, 7, mb.Models["mistral"].Priority)

	_, found := mb.Models["llama"]
	assert.False(t, found)
}
