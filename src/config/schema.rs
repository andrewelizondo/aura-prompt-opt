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
}

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
