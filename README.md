# aura-prompt-opt

A DSPy-style prompt optimizer for [Aura](https://github.com/mezmo/aura) TOML agent configurations.

Takes an Aura TOML config, runs it against evaluation scenarios, iteratively optimizes the system prompt and config parameters, and outputs:
- `config.optimized.toml` — production-ready minimal TOML
- `config.verbose.toml` — annotated TOML explaining every optimization decision

## Quick Start

```bash
# Set your LLM API key (any OpenAI-compatible endpoint)
export OPTIMIZER_API_KEY=sk-...

# Run the SRE agent example
cargo run --example optimize_sre_agent
```

## Configuration

| Env Var | Default | Description |
|---|---|---|
| `OPTIMIZER_API_KEY` | — | API key (falls back to `OPENAI_API_KEY`) |
| `OPTIMIZER_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `OPTIMIZER_MODEL` | `gpt-4o` | Model for optimization calls |

## Usage

```rust
use aura_prompt_opt::{AuraOptimizer, EvalDataset, EvalScenario, FuzzyMatch};

let dataset = EvalDataset::new("my-eval")
    .add(EvalScenario::new("incident", "pods are crashing in production")
        .with_task_description("Agent should diagnose the incident"));

let metrics = vec![Box::new(FuzzyMatch::new()) as Box<dyn aura_prompt_opt::Metric>];
let optimizer = AuraOptimizer::from_env(metrics);

let output = optimizer.optimize_file(
    std::path::Path::new("config.toml"),
    &dataset,
).await?;

std::fs::write("config.optimized.toml", &output.optimized_toml)?;
std::fs::write("config.verbose.toml", &output.verbose_toml)?;
```

## DSPy Concepts Implemented

- **BootstrapFewShot** — collects passing eval examples and injects them as few-shot demonstrations into the system prompt
- **InstructionOptimizer** — uses an LLM to analyze failures and propose improved prompt instructions
- **MiproOptimizer** — runs multiple rounds of instruction optimization, picking the best result each round
- **Metrics** — `ExactMatch`, `FuzzyMatch`, `LlmJudge` (all implement the pluggable `Metric` trait)

## Verbose Output Example

```toml
# ============================================================
# aura-prompt-opt — Verbose Optimization Report
# ============================================================
# Baseline score:  0.450
# Optimized score: 0.820
# Score delta:     +0.370
# Optimizations:   1
# ============================================================

# OPTIMIZED by BootstrapFewShot: agent.system_prompt
# Before score: 0.450 → After score: 0.820 (+0.370)
# Confidence: HIGH
# Rationale: Injected 3 few-shot examples (score >= 0.70) from baseline evaluation.
# Alternatives tried:
#   score=0.670: "You are a Kubernetes SRE. Use tools systematically..."
#   score=0.720: "You are a senior SRE engineer..."
#
[agent]
name = "Kubernetes SRE Agent"
system_prompt = """
You are a senior Site Reliability Engineer...
"""
```

## Architecture

```
src/
├── config/       # Parse/serialize Aura TOML configs
├── llm/          # OpenAI-compatible async HTTP client
├── eval/         # Scenarios, metrics, runner
├── optimizer/    # BootstrapFewShot, InstructionOptimizer, MiproOptimizer
└── compiler/     # Emit optimized or verbose-annotated TOML
```

## License

Apache-2.0
