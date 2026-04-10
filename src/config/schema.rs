use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

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
    #[serde(default)] pub alias: Option<String>,
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
            alias: None,
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
    /// Workers keyed by name. Accepts both `[orchestration.workers.X]` (plural)
    /// and `[orchestration.worker.X]` (singular, as used by Aura).
    #[serde(default)] pub workers: HashMap<String, WorkerConfig>,
    /// Alias: Aura uses `[orchestration.worker.X]` (singular).
    #[serde(default, alias = "worker")] pub worker: Option<HashMap<String, WorkerConfig>>,
    #[serde(default)] pub prompts: OrchestrationPrompts,
    #[serde(default)] pub allow_direct_answers: Option<bool>,
    #[serde(default)] pub allow_clarification: Option<bool>,
    #[serde(default)] pub tools_in_planning: Option<String>,
    #[serde(default)] pub timeouts: Option<TimeoutsConfig>,
    #[serde(default)] pub artifacts: Option<ArtifactsConfig>,
}

impl OrchestrationConfig {
    /// Returns all workers, merging `workers` (plural) and `worker` (singular) maps.
    pub fn all_workers(&self) -> HashMap<String, WorkerConfig> {
        let mut merged = self.workers.clone();
        if let Some(ref singular) = self.worker {
            for (name, cfg) in singular {
                merged.entry(name.clone()).or_insert_with(|| cfg.clone());
            }
        }
        merged
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TimeoutsConfig {
    #[serde(default)] pub per_call_timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ArtifactsConfig {
    #[serde(default)] pub memory_dir: Option<String>,
}

/// Overridable prompt templates for orchestration mode.
///
/// Backed by a `BTreeMap<String, String>` so new prompts added to Aura
/// are picked up automatically without code changes. Keys are the prompt
/// name (e.g. `"synthesis_prompt"`), values are the full template text.
///
/// Defaults are populated from the embedded constants in `crate::prompts`.
/// When deserializing from TOML, any user-provided overrides are merged on
/// top of the embedded defaults, so unspecified prompts keep their defaults
/// and new prompts discovered at runtime are included.
#[derive(Debug, Clone, PartialEq)]
pub struct OrchestrationPrompts {
    pub prompts: BTreeMap<String, String>,
}

impl Default for OrchestrationPrompts {
    fn default() -> Self {
        Self { prompts: crate::prompts::embedded_defaults() }
    }
}

impl Serialize for OrchestrationPrompts {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.prompts.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for OrchestrationPrompts {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let user_overrides = BTreeMap::<String, String>::deserialize(deserializer)?;
        // Start from embedded defaults, then overlay user overrides
        let mut merged = crate::prompts::embedded_defaults();
        for (key, value) in user_overrides {
            merged.insert(key, value);
        }
        Ok(OrchestrationPrompts { prompts: merged })
    }
}

impl OrchestrationPrompts {
    /// Creates prompts from an Aura repo checkout, with embedded defaults as fallback.
    pub fn from_aura_repo(aura_repo_root: &std::path::Path) -> crate::Result<Self> {
        let discovered = crate::prompts::discover_from_aura_repo(aura_repo_root)?;
        let discovered_map = crate::prompts::discovered_to_map(&discovered);
        // Discovered prompts override embedded defaults
        let merged = crate::prompts::merge_prompt_maps(
            &crate::prompts::embedded_defaults(),
            &discovered_map,
        );
        Ok(Self { prompts: merged })
    }

    /// Creates prompts by discovering `.md` files from a directory.
    pub fn from_prompt_dir(dir: &std::path::Path) -> crate::Result<Self> {
        let discovered = crate::prompts::discover_from_dir(dir)?;
        let discovered_map = crate::prompts::discovered_to_map(&discovered);
        let merged = crate::prompts::merge_prompt_maps(
            &crate::prompts::embedded_defaults(),
            &discovered_map,
        );
        Ok(Self { prompts: merged })
    }

    /// Returns all prompt entries as (dotted_field_path, prompt_value) pairs.
    pub fn fields(&self) -> Vec<(String, &str)> {
        self.prompts
            .iter()
            .map(|(name, content)| {
                (format!("orchestration.prompts.{name}"), content.as_str())
            })
            .collect()
    }

    /// Sets a prompt field by its dotted path. Returns true if successful.
    ///
    /// Accepts both `"orchestration.prompts.synthesis_prompt"` and bare `"synthesis_prompt"`.
    pub fn set_field(&mut self, field: &str, value: String) -> bool {
        let key = field
            .strip_prefix("orchestration.prompts.")
            .unwrap_or(field);
        self.prompts.insert(key.to_string(), value);
        true
    }

    /// Gets a prompt field value by its dotted path or bare name.
    pub fn get_field(&self, field: &str) -> Option<&str> {
        let key = field
            .strip_prefix("orchestration.prompts.")
            .unwrap_or(field);
        self.prompts.get(key).map(|s| s.as_str())
    }

    /// Returns the number of prompt templates.
    pub fn len(&self) -> usize {
        self.prompts.len()
    }

    /// Returns true if there are no prompt templates.
    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }
}

fn default_max_planning_cycles() -> usize { 3 }
fn default_quality_threshold() -> f32 { 0.8 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerConfig {
    pub description: String,
    #[serde(default)] pub preamble: String,
    #[serde(default)] pub turn_depth: Option<usize>,
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
