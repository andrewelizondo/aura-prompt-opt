// placeholder
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct AuraConfig {
    pub llm: LlmConfig,
    #[serde(default)]
    pub mcp: Option<McpConfig>,
    #[serde(default)]
    pub vector_stores: Vec<VectorStoreConfig>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
    pub agent: AgentConfig,
    #[serde(default)]
    pub orchestration: Option<OrchestrationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum LlmConfig {
    OpenAI { api_key: String, model: String, #[serde(default)] base_url: Option<String> },
    Anthropic { api_key: String, model: String, #[serde(default)] base_url: Option<String> },
    Ollama { model: String, #[serde(default = "default_ollama_base_url")] base_url: String },
    Bedrock { model: String, region: String, #[serde(default)] profile: Option<String> },
    Gemini { api_key: String, model: String, #[serde(default)] base_url: Option<String> },
}

fn default_ollama_base_url() -> String { "http://localhost:11434".to_string() }

impl Default for LlmConfig {
    fn default() -> Self {
        LlmConfig::OpenAI { api_key: String::new(), model: "gpt-4o".to_string(), base_url: None }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct McpConfig {
    pub servers: HashMap<String, McpServerConfig>,
    #[serde(default = "default_sanitize_schemas")]
    pub sanitize_schemas: bool,
}

fn default_sanitize_schemas() -> bool { true }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "transport")]
pub enum McpServerConfig {
    #[serde(rename = "http_streamable")]
    HttpStreamable {
        url: String,
        #[serde(default)] headers: HashMap<String, String>,
        #[serde(default)] description: Option<String>,
        #[serde(default)] headers_from_request: HashMap<String, String>,
    },
    #[serde(rename = "stdio")]
    Stdio {
        cmd: Vec<String>,
        #[serde(default)] args: Vec<String>,
        #[serde(default)] env: HashMap<String, String>,
        #[serde(default)] description: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorStoreConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub store_type: String,
    pub embedding_model: EmbeddingConfig,
    #[serde(default)] pub url: Option<String>,
    #[serde(default)] pub collection_name: Option<String>,
    #[serde(default)] pub context_prefix: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub model: String,
    pub api_key: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ToolsConfig {
    #[serde(default)] pub filesystem: bool,
    #[serde(default)] pub custom_tools: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentConfig {
    pub name: String,
    pub system_prompt: String,
    #[serde(default)] pub temperature: Option<f64>,
    #[serde(default)] pub max_tokens: Option<u64>,
    #[serde(default = "default_turn_depth")] pub turn_depth: Option<usize>,
    #[serde(default)] pub mcp_filter: Option<Vec<String>>,
}

fn default_turn_depth() -> Option<usize> { Some(5) }

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "Assistant".to_string(),
            system_prompt: "You are a helpful assistant.".to_string(),
            temperature: Some(0.7),
            max_tokens: None,
            turn_depth: default_turn_depth(),
            mcp_filter: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct OrchestrationConfig {
    #[serde(default)] pub enabled: bool,
    #[serde(default = "default_max_planning_cycles")] pub max_planning_cycles: usize,
    #[serde(default = "default_quality_threshold")] pub quality_threshold: f32,
    #[serde(default)] pub workers: HashMap<String, WorkerConfig>,
    #[serde(default)] pub prompts: OrchestrationPrompts,
}

/// Overridable prompt templates for orchestration mode.
///
/// Each field defaults to the built-in prompt from Aura's orchestration mode.
/// The optimizer populates these with improved variants when optimizing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrchestrationPrompts {
    #[serde(default = "default_orchestrator_preamble")]
    pub orchestrator_preamble: String,
    #[serde(default = "default_worker_preamble")]
    pub worker_preamble: String,
    #[serde(default = "default_worker_task_prompt")]
    pub worker_task_prompt: String,
    #[serde(default = "default_synthesis_prompt")]
    pub synthesis_prompt: String,
    #[serde(default = "default_evaluation_preamble")]
    pub evaluation_preamble: String,
    #[serde(default = "default_evaluation_prompt")]
    pub evaluation_prompt: String,
    #[serde(default = "default_reflection_prompt")]
    pub reflection_prompt: String,
    #[serde(default = "default_phase_continuation_prompt")]
    pub phase_continuation_prompt: String,
    #[serde(default = "default_session_history_template")]
    pub session_history_template: String,
    #[serde(default = "default_todo_system_prompt")]
    pub todo_system_prompt: String,
    #[serde(default = "default_todo_tool_prompt")]
    pub todo_tool_prompt: String,
}

impl Default for OrchestrationPrompts {
    fn default() -> Self {
        Self {
            orchestrator_preamble: default_orchestrator_preamble(),
            worker_preamble: default_worker_preamble(),
            worker_task_prompt: default_worker_task_prompt(),
            synthesis_prompt: default_synthesis_prompt(),
            evaluation_preamble: default_evaluation_preamble(),
            evaluation_prompt: default_evaluation_prompt(),
            reflection_prompt: default_reflection_prompt(),
            phase_continuation_prompt: default_phase_continuation_prompt(),
            session_history_template: default_session_history_template(),
            todo_system_prompt: default_todo_system_prompt(),
            todo_tool_prompt: default_todo_tool_prompt(),
        }
    }
}

impl OrchestrationPrompts {
    /// Returns an iterator over (field_path, prompt_value) for all prompt fields.
    pub fn fields(&self) -> Vec<(&'static str, &str)> {
        vec![
            ("orchestration.prompts.orchestrator_preamble", &self.orchestrator_preamble),
            ("orchestration.prompts.worker_preamble", &self.worker_preamble),
            ("orchestration.prompts.worker_task_prompt", &self.worker_task_prompt),
            ("orchestration.prompts.synthesis_prompt", &self.synthesis_prompt),
            ("orchestration.prompts.evaluation_preamble", &self.evaluation_preamble),
            ("orchestration.prompts.evaluation_prompt", &self.evaluation_prompt),
            ("orchestration.prompts.reflection_prompt", &self.reflection_prompt),
            ("orchestration.prompts.phase_continuation_prompt", &self.phase_continuation_prompt),
            ("orchestration.prompts.session_history_template", &self.session_history_template),
            ("orchestration.prompts.todo_system_prompt", &self.todo_system_prompt),
            ("orchestration.prompts.todo_tool_prompt", &self.todo_tool_prompt),
        ]
    }

    /// Sets a prompt field by its dotted path. Returns true if the field was found.
    pub fn set_field(&mut self, field: &str, value: String) -> bool {
        match field {
            "orchestration.prompts.orchestrator_preamble" => self.orchestrator_preamble = value,
            "orchestration.prompts.worker_preamble" => self.worker_preamble = value,
            "orchestration.prompts.worker_task_prompt" => self.worker_task_prompt = value,
            "orchestration.prompts.synthesis_prompt" => self.synthesis_prompt = value,
            "orchestration.prompts.evaluation_preamble" => self.evaluation_preamble = value,
            "orchestration.prompts.evaluation_prompt" => self.evaluation_prompt = value,
            "orchestration.prompts.reflection_prompt" => self.reflection_prompt = value,
            "orchestration.prompts.phase_continuation_prompt" => self.phase_continuation_prompt = value,
            "orchestration.prompts.session_history_template" => self.session_history_template = value,
            "orchestration.prompts.todo_system_prompt" => self.todo_system_prompt = value,
            "orchestration.prompts.todo_tool_prompt" => self.todo_tool_prompt = value,
            _ => return false,
        }
        true
    }

    /// Gets a prompt field value by its dotted path.
    pub fn get_field(&self, field: &str) -> Option<&str> {
        match field {
            "orchestration.prompts.orchestrator_preamble" => Some(&self.orchestrator_preamble),
            "orchestration.prompts.worker_preamble" => Some(&self.worker_preamble),
            "orchestration.prompts.worker_task_prompt" => Some(&self.worker_task_prompt),
            "orchestration.prompts.synthesis_prompt" => Some(&self.synthesis_prompt),
            "orchestration.prompts.evaluation_preamble" => Some(&self.evaluation_preamble),
            "orchestration.prompts.evaluation_prompt" => Some(&self.evaluation_prompt),
            "orchestration.prompts.reflection_prompt" => Some(&self.reflection_prompt),
            "orchestration.prompts.phase_continuation_prompt" => Some(&self.phase_continuation_prompt),
            "orchestration.prompts.session_history_template" => Some(&self.session_history_template),
            "orchestration.prompts.todo_system_prompt" => Some(&self.todo_system_prompt),
            "orchestration.prompts.todo_tool_prompt" => Some(&self.todo_tool_prompt),
            _ => None,
        }
    }
}

fn default_orchestrator_preamble() -> String { crate::prompts::ORCHESTRATOR_PREAMBLE.to_string() }
fn default_worker_preamble() -> String { crate::prompts::WORKER_PREAMBLE.to_string() }
fn default_worker_task_prompt() -> String { crate::prompts::WORKER_TASK_PROMPT.to_string() }
fn default_synthesis_prompt() -> String { crate::prompts::SYNTHESIS_PROMPT.to_string() }
fn default_evaluation_preamble() -> String { crate::prompts::EVALUATION_PREAMBLE.to_string() }
fn default_evaluation_prompt() -> String { crate::prompts::EVALUATION_PROMPT.to_string() }
fn default_reflection_prompt() -> String { crate::prompts::REFLECTION_PROMPT.to_string() }
fn default_phase_continuation_prompt() -> String { crate::prompts::PHASE_CONTINUATION_PROMPT.to_string() }
fn default_session_history_template() -> String { crate::prompts::SESSION_HISTORY_TEMPLATE.to_string() }
fn default_todo_system_prompt() -> String { crate::prompts::TODO_SYSTEM_PROMPT.to_string() }
fn default_todo_tool_prompt() -> String { crate::prompts::TODO_TOOL_PROMPT.to_string() }

fn default_max_planning_cycles() -> usize { 3 }
fn default_quality_threshold() -> f32 { 0.8 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerConfig {
    pub description: String,
    pub preamble: String,
    #[serde(default)] pub mcp_filter: Vec<String>,
}

impl LlmConfig {
    pub fn provider_name(&self) -> &str {
        match self {
            LlmConfig::OpenAI { .. } => "openai",
            LlmConfig::Anthropic { .. } => "anthropic",
            LlmConfig::Ollama { .. } => "ollama",
            LlmConfig::Bedrock { .. } => "bedrock",
            LlmConfig::Gemini { .. } => "gemini",
        }
    }

    pub fn model_name(&self) -> &str {
        match self {
            LlmConfig::OpenAI { model, .. } => model,
            LlmConfig::Anthropic { model, .. } => model,
            LlmConfig::Ollama { model, .. } => model,
            LlmConfig::Bedrock { model, .. } => model,
            LlmConfig::Gemini { model, .. } => model,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_config_provider_name() {
        let cfg = LlmConfig::OpenAI { api_key: "sk-test".into(), model: "gpt-4o".into(), base_url: None };
        assert_eq!(cfg.provider_name(), "openai");
        assert_eq!(cfg.model_name(), "gpt-4o");
    }

    #[test]
    fn test_agent_config_default() {
        let cfg = AgentConfig::default();
        assert_eq!(cfg.name, "Assistant");
        assert_eq!(cfg.turn_depth, Some(5));
    }
}
