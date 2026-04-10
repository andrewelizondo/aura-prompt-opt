pub mod bootstrap;
pub mod instruction;
pub mod mipro;

pub use bootstrap::BootstrapFewShot;
pub use instruction::InstructionOptimizer;
pub use mipro::MiproOptimizer;

use crate::config::schema::AuraConfig;
use crate::eval::EvalResult;

/// Identifies a prompt field in the config that can be targeted for optimization.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OptimizableField {
    /// The agent's top-level system prompt (`agent.system_prompt`).
    AgentSystemPrompt,
    /// An orchestration prompt template, identified by its dotted config path.
    OrchestrationPrompt(String),
    /// A worker's preamble, identified by worker name.
    WorkerPreamble(String),
}

impl OptimizableField {
    /// Returns the dotted config path for this field.
    pub fn field_path(&self) -> String {
        match self {
            Self::AgentSystemPrompt => "agent.system_prompt".to_string(),
            Self::OrchestrationPrompt(name) => name.clone(),
            Self::WorkerPreamble(name) => format!("orchestration.workers.{name}.preamble"),
        }
    }

    /// Reads the current value of this field from the config.
    pub fn get_value<'a>(&self, config: &'a AuraConfig) -> Option<&'a str> {
        match self {
            Self::AgentSystemPrompt => Some(&config.agent.system_prompt),
            Self::OrchestrationPrompt(path) => {
                config.orchestration.as_ref()
                    .and_then(|o| o.prompts.get_field(path))
            }
            Self::WorkerPreamble(name) => {
                config.orchestration.as_ref()
                    .and_then(|o| o.workers.get(name))
                    .map(|w| w.preamble.as_str())
            }
        }
    }

    /// Sets the value of this field on the config. Returns true if successful.
    pub fn set_value(&self, config: &mut AuraConfig, value: String) -> bool {
        match self {
            Self::AgentSystemPrompt => {
                config.agent.system_prompt = value;
                true
            }
            Self::OrchestrationPrompt(path) => {
                if let Some(ref mut orch) = config.orchestration {
                    orch.prompts.set_field(path, value)
                } else {
                    false
                }
            }
            Self::WorkerPreamble(name) => {
                if let Some(ref mut orch) = config.orchestration {
                    if let Some(worker) = orch.workers.get_mut(name) {
                        worker.preamble = value;
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Returns all optimizable fields present in the given config.
    pub fn all_fields(config: &AuraConfig) -> Vec<OptimizableField> {
        let mut fields = vec![OptimizableField::AgentSystemPrompt];

        if let Some(ref orch) = config.orchestration {
            if orch.enabled {
                // Add all orchestration prompt fields
                for (path, _) in orch.prompts.fields() {
                    fields.push(OptimizableField::OrchestrationPrompt(path.to_string()));
                }
                // Add worker preambles
                for name in orch.workers.keys() {
                    fields.push(OptimizableField::WorkerPreamble(name.clone()));
                }
            }
        }

        fields
    }

    /// Returns a human-readable description of what this prompt does.
    pub fn description(&self) -> &str {
        match self {
            Self::AgentSystemPrompt => "The agent's main system prompt",
            Self::OrchestrationPrompt(path) => match path.as_str() {
                "orchestration.prompts.orchestrator_preamble" =>
                    "Coordinator agent preamble for routing and planning",
                "orchestration.prompts.worker_preamble" =>
                    "Worker agent preamble defining scope and rules",
                "orchestration.prompts.worker_task_prompt" =>
                    "Template injected per-task into worker agents",
                "orchestration.prompts.synthesis_prompt" =>
                    "Prompt for combining multi-task results into a response",
                "orchestration.prompts.evaluation_preamble" =>
                    "Evaluation agent system prompt",
                "orchestration.prompts.evaluation_prompt" =>
                    "Quality assessment prompt with scoring criteria",
                "orchestration.prompts.reflection_prompt" =>
                    "Replan cycle prompt for iterative improvement",
                "orchestration.prompts.phase_continuation_prompt" =>
                    "Decision prompt for continue vs replan between phases",
                "orchestration.prompts.session_history_template" =>
                    "Multi-turn session context injection template",
                "orchestration.prompts.todo_system_prompt" =>
                    "System-level task management guidance",
                "orchestration.prompts.todo_tool_prompt" =>
                    "Todo tool usage instructions for agents",
                _ => "Orchestration prompt template",
            },
            Self::WorkerPreamble(_) => "A specialized worker agent preamble",
        }
    }

    /// Returns true if this is a template prompt (contains %%VAR%% or {{var}} placeholders).
    pub fn is_template(&self) -> bool {
        !matches!(self, Self::AgentSystemPrompt)
    }
}

/// Computes the average aggregate score across a slice of eval results.
pub fn average_score(results: &[EvalResult]) -> f64 {
    if results.is_empty() { return 0.0; }
    results.iter().map(|r| r.aggregate_score).sum::<f64>() / results.len() as f64
}

/// Strips markdown code fences from LLM JSON responses.
pub fn strip_code_fences(s: &str) -> &str {
    let s = s.trim();
    let s = s.strip_prefix("```json").unwrap_or(s);
    let s = s.strip_prefix("```").unwrap_or(s);
    let s = s.strip_suffix("```").unwrap_or(s);
    s.trim()
}

/// A candidate config with its evaluation score.
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    pub config: AuraConfig,
    pub score: f64,
    pub results: Vec<EvalResult>,
    pub notes: Vec<String>,
}

/// The full output of an optimization run.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub best_config: AuraConfig,
    pub best_score: f64,
    pub candidates: Vec<ScoredCandidate>,
    pub baseline_score: f64,
    pub optimization_log: Vec<OptimizationLogEntry>,
}

/// A single optimization decision recorded for the verbose output.
#[derive(Debug, Clone)]
pub struct OptimizationLogEntry {
    pub optimizer: String,
    pub field: String,
    pub before: String,
    pub after: String,
    pub score_before: f64,
    pub score_after: f64,
    pub rationale: String,
    pub alternatives: Vec<(String, f64)>,
}
