/// Extract and inject TOML configs from Helm values YAML files.
///
/// Handles the pattern where an Aura TOML config is embedded inside a
/// YAML values file under a key like `config.content: |`.

use crate::error::{Error, Result};
use serde_yaml::Value as YamlValue;

/// Extracts TOML content from a YAML values file at the given dotted key path.
///
/// For example, with `key_path = "config.content"` and a YAML file containing:
/// ```yaml
/// config:
///   content: |
///     [llm]
///     provider = "openai"
///     ...
/// ```
/// This returns the TOML string.
pub fn extract_toml_from_yaml(yaml_str: &str, key_path: &str) -> Result<String> {
    let yaml: YamlValue = serde_yaml::from_str(yaml_str)
        .map_err(|e| Error::Config(format!("failed to parse YAML: {e}")))?;

    let mut current = &yaml;
    for key in key_path.split('.') {
        current = current.get(key).ok_or_else(|| {
            Error::Config(format!("key '{key}' not found in YAML (path: {key_path})"))
        })?;
    }

    current
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| Error::Config(format!(
            "value at '{key_path}' is not a string (found {:?})",
            yaml_type_name(current)
        )))
}

/// Injects TOML content back into a YAML values file at the given dotted key path.
///
/// Preserves the rest of the YAML structure, only replacing the value at the path.
pub fn inject_toml_into_yaml(yaml_str: &str, key_path: &str, toml_content: &str) -> Result<String> {
    let mut yaml: YamlValue = serde_yaml::from_str(yaml_str)
        .map_err(|e| Error::Config(format!("failed to parse YAML: {e}")))?;

    let keys: Vec<&str> = key_path.split('.').collect();
    set_nested_yaml_value(&mut yaml, &keys, toml_content)?;

    serde_yaml::to_string(&yaml)
        .map_err(|e| Error::Config(format!("failed to serialize YAML: {e}")))
}

fn set_nested_yaml_value(yaml: &mut YamlValue, keys: &[&str], value: &str) -> Result<()> {
    if keys.is_empty() {
        return Err(Error::Config("empty key path".into()));
    }

    if keys.len() == 1 {
        if let YamlValue::Mapping(ref mut map) = yaml {
            map.insert(
                YamlValue::String(keys[0].to_string()),
                YamlValue::String(value.to_string()),
            );
            return Ok(());
        }
        return Err(Error::Config(format!("cannot set key '{}' on non-mapping", keys[0])));
    }

    if let YamlValue::Mapping(ref mut map) = yaml {
        let key = YamlValue::String(keys[0].to_string());
        let entry = map.entry(key).or_insert(YamlValue::Mapping(serde_yaml::Mapping::new()));
        set_nested_yaml_value(entry, &keys[1..], value)
    } else {
        Err(Error::Config(format!("cannot traverse key '{}' on non-mapping", keys[0])))
    }
}

fn yaml_type_name(v: &YamlValue) -> &'static str {
    match v {
        YamlValue::Null => "null",
        YamlValue::Bool(_) => "bool",
        YamlValue::Number(_) => "number",
        YamlValue::String(_) => "string",
        YamlValue::Sequence(_) => "array",
        YamlValue::Mapping(_) => "object",
        YamlValue::Tagged(_) => "tagged",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_VALUES: &str = r#"
replicaCount: 1
image: mezmo/aura:orchestration
config:
  content: |
    [llm]
    provider = "openai"
    api_key = "{{ env.OPENAI_API_KEY }}"
    model = "gpt-4o"

    [agent]
    name = "Test Agent"
    system_prompt = "You are helpful."
server:
  httpPort: 8080
"#;

    #[test]
    fn test_extract_toml_from_yaml() {
        let toml = extract_toml_from_yaml(SAMPLE_VALUES, "config.content").unwrap();
        assert!(toml.contains("[llm]"));
        assert!(toml.contains("[agent]"));
        assert!(toml.contains("gpt-4o"));
    }

    #[test]
    fn test_extract_missing_key() {
        let result = extract_toml_from_yaml(SAMPLE_VALUES, "config.nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("nonexistent"));
    }

    #[test]
    fn test_extract_non_string_value() {
        let result = extract_toml_from_yaml(SAMPLE_VALUES, "replicaCount");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a string"));
    }

    #[test]
    fn test_inject_toml_into_yaml() {
        let new_toml = "[llm]\nprovider = \"anthropic\"\n";
        let result = inject_toml_into_yaml(SAMPLE_VALUES, "config.content", new_toml).unwrap();

        // The injected TOML should be in the output
        let extracted = extract_toml_from_yaml(&result, "config.content").unwrap();
        assert!(extracted.contains("anthropic"));

        // Other YAML keys should be preserved
        assert!(result.contains("replicaCount"));
        assert!(result.contains("httpPort"));
    }

    #[test]
    fn test_round_trip_preserves_structure() {
        let toml = extract_toml_from_yaml(SAMPLE_VALUES, "config.content").unwrap();
        let result = inject_toml_into_yaml(SAMPLE_VALUES, "config.content", &toml).unwrap();
        let re_extracted = extract_toml_from_yaml(&result, "config.content").unwrap();
        assert_eq!(toml.trim(), re_extracted.trim());
    }
}
