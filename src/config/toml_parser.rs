use crate::error::Result;
use super::schema::AuraConfig;

pub fn parse_toml(input: &str) -> Result<AuraConfig> {
    let config: AuraConfig = toml::from_str(input)?;
    Ok(config)
}

pub fn serialize_toml(config: &AuraConfig) -> Result<String> {
    let s = toml::to_string_pretty(config)?;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_TOML: &str = r#"
[llm]
provider = "openai"
api_key = "sk-test"
model = "gpt-4o"

[agent]
name = "Test Agent"
system_prompt = "You are a helpful assistant."
"#;

    #[test]
    fn test_parse_minimal_config() {
        let cfg = parse_toml(MINIMAL_TOML).unwrap();
        assert_eq!(cfg.agent.name, "Test Agent");
        assert_eq!(cfg.agent.system_prompt, "You are a helpful assistant.");
        assert!(cfg.mcp.is_none());
    }

    #[test]
    fn test_parse_config_with_mcp() {
        let toml_str = r#"
[llm]
provider = "openai"
api_key = "sk-test"
model = "gpt-4o"

[mcp]
sanitize_schemas = true

[mcp.servers.kubernetes]
transport = "http_streamable"
url = "http://localhost:8081/mcp"
description = "Kubernetes cluster operations"

[agent]
name = "SRE Agent"
system_prompt = "You are an SRE."
turn_depth = 20
"#;
        let cfg = parse_toml(toml_str).unwrap();
        let mcp = cfg.mcp.as_ref().unwrap();
        assert!(mcp.servers.contains_key("kubernetes"));
        assert_eq!(cfg.agent.turn_depth, Some(20));
    }

    #[test]
    fn test_round_trip() {
        let cfg = parse_toml(MINIMAL_TOML).unwrap();
        let serialized = serialize_toml(&cfg).unwrap();
        let reparsed = parse_toml(&serialized).unwrap();
        assert_eq!(cfg.agent.name, reparsed.agent.name);
        assert_eq!(cfg.agent.system_prompt, reparsed.agent.system_prompt);
    }

    #[test]
    fn test_parse_error_on_invalid_toml() {
        let result = parse_toml("not valid toml [[[");
        assert!(result.is_err());
    }
}
