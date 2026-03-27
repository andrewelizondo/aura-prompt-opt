use crate::config::schema::AuraConfig;
use crate::error::Result;

pub fn compile_optimized(config: &AuraConfig) -> Result<String> {
    let toml_str = toml::to_string_pretty(config)?;
    Ok(toml_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::{AgentConfig, AuraConfig, LlmConfig};

    #[test]
    fn test_compile_optimized_produces_valid_toml() {
        let config = AuraConfig {
            llm: LlmConfig::OpenAI { api_key: "sk-test".into(), model: "gpt-4o".into(), base_url: None },
            agent: AgentConfig {
                name: "MyAgent".into(),
                system_prompt: "You are helpful.".into(),
                ..AgentConfig::default()
            },
            ..AuraConfig::default()
        };

        let toml_str = compile_optimized(&config).unwrap();
        let reparsed: AuraConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(reparsed.agent.name, "MyAgent");
        assert_eq!(reparsed.agent.system_prompt, "You are helpful.");
    }
}
