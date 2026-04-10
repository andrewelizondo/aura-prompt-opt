/// Prompt discovery and management for Aura orchestration templates.
///
/// Supports three layers of prompt resolution:
/// 1. **Embedded defaults** — compiled-in from Aura's `feature/orchestration-mode` branch
/// 2. **Discovered from Aura repo** — reads `.md` files from a local checkout (dynamic)
/// 3. **User TOML overrides** — individual prompt overrides in the config file
///
/// Each layer overrides the previous.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// ── Embedded defaults ──────────────────────────────────���───────────────

pub const ORCHESTRATOR_PREAMBLE: &str = include_str!("prompts/orchestrator_preamble.md");
pub const WORKER_PREAMBLE: &str = include_str!("prompts/worker_preamble.md");
pub const WORKER_TASK_PROMPT: &str = include_str!("prompts/worker_task_prompt.md");
pub const SYNTHESIS_PROMPT: &str = include_str!("prompts/synthesis_prompt.md");
pub const EVALUATION_PREAMBLE: &str = include_str!("prompts/evaluation_preamble.md");
pub const EVALUATION_PROMPT: &str = include_str!("prompts/evaluation_prompt.md");
pub const REFLECTION_PROMPT: &str = include_str!("prompts/reflection_prompt.md");
pub const PHASE_CONTINUATION_PROMPT: &str = include_str!("prompts/phase_continuation_prompt.md");
pub const SESSION_HISTORY_TEMPLATE: &str = include_str!("prompts/session_history_template.md");
pub const TODO_SYSTEM_PROMPT: &str = include_str!("prompts/todo_system_prompt.md");
pub const TODO_TOOL_PROMPT: &str = include_str!("prompts/todo_tool_prompt.md");

/// Returns the embedded prompt defaults as a map of name → content.
pub fn embedded_defaults() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("orchestrator_preamble".into(), ORCHESTRATOR_PREAMBLE.into()),
        ("worker_preamble".into(), WORKER_PREAMBLE.into()),
        ("worker_task_prompt".into(), WORKER_TASK_PROMPT.into()),
        ("synthesis_prompt".into(), SYNTHESIS_PROMPT.into()),
        ("evaluation_preamble".into(), EVALUATION_PREAMBLE.into()),
        ("evaluation_prompt".into(), EVALUATION_PROMPT.into()),
        ("reflection_prompt".into(), REFLECTION_PROMPT.into()),
        ("phase_continuation_prompt".into(), PHASE_CONTINUATION_PROMPT.into()),
        ("session_history_template".into(), SESSION_HISTORY_TEMPLATE.into()),
        ("todo_system_prompt".into(), TODO_SYSTEM_PROMPT.into()),
        ("todo_tool_prompt".into(), TODO_TOOL_PROMPT.into()),
    ])
}

// ── Dynamic discovery ─────────────────────────────���────────────────────

/// A prompt discovered from an Aura repo checkout.
#[derive(Debug, Clone)]
pub struct DiscoveredPrompt {
    /// Short name derived from the filename stem (e.g. `synthesis_prompt`).
    pub name: String,
    /// Full content of the `.md` file.
    pub content: String,
    /// Template variables found in the content (`%%VAR%%` and `{{var}}`).
    pub template_vars: Vec<String>,
    /// Source file path.
    pub source_path: PathBuf,
}

/// Files to skip when scanning the prompts directory.
const SKIP_FILES: &[&str] = &["mod.rs", "templates.md"];

/// Discovers prompt templates from a local Aura repository checkout.
///
/// Scans `{aura_repo_root}/crates/aura/src/prompts/` for `.md` files,
/// reads their content, and extracts template variables.
///
/// # Arguments
/// * `aura_repo_root` — Path to the root of an Aura git checkout.
///
/// # Returns
/// A vector of discovered prompts. Returns an error if the prompts directory
/// does not exist or cannot be read.
pub fn discover_from_aura_repo(aura_repo_root: &Path) -> crate::Result<Vec<DiscoveredPrompt>> {
    let prompts_dir = aura_repo_root.join("crates/aura/src/prompts");
    discover_from_dir(&prompts_dir)
}

/// Discovers prompt templates from a directory of `.md` files.
///
/// This is the lower-level function — use `discover_from_aura_repo` if you
/// have the repo root, or call this directly with a custom directory.
pub fn discover_from_dir(dir: &Path) -> crate::Result<Vec<DiscoveredPrompt>> {
    if !dir.is_dir() {
        return Err(crate::Error::Config(format!(
            "prompts directory not found: {}",
            dir.display()
        )));
    }

    let mut prompts = Vec::new();

    let entries = std::fs::read_dir(dir).map_err(|e| {
        crate::Error::Config(format!("failed to read prompts directory {}: {e}", dir.display()))
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| {
            crate::Error::Config(format!("failed to read directory entry: {e}"))
        })?;

        let path = entry.path();
        let file_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name.to_string(),
            None => continue,
        };

        // Skip non-.md files and known non-prompt files
        if !file_name.ends_with(".md") || SKIP_FILES.contains(&file_name.as_str()) {
            continue;
        }

        let stem = file_name.strip_suffix(".md").unwrap_or(&file_name);
        let content = std::fs::read_to_string(&path).map_err(|e| {
            crate::Error::Config(format!("failed to read {}: {e}", path.display()))
        })?;

        let template_vars = extract_template_vars(&content);

        prompts.push(DiscoveredPrompt {
            name: stem.to_string(),
            content,
            template_vars,
            source_path: path,
        });
    }

    // Sort by name for deterministic ordering
    prompts.sort_by(|a, b| a.name.cmp(&b.name));

    tracing::info!(
        "Discovered {} prompt templates from {}",
        prompts.len(),
        dir.display()
    );

    Ok(prompts)
}

/// Converts discovered prompts into a BTreeMap suitable for `OrchestrationPrompts`.
pub fn discovered_to_map(prompts: &[DiscoveredPrompt]) -> BTreeMap<String, String> {
    prompts.iter().map(|p| (p.name.clone(), p.content.clone())).collect()
}

/// Merges prompt maps with `overrides` taking precedence over `base`.
pub fn merge_prompt_maps(
    base: &BTreeMap<String, String>,
    overrides: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut merged = base.clone();
    for (key, value) in overrides {
        merged.insert(key.clone(), value.clone());
    }
    merged
}

/// Extracts all template variables (`%%VAR%%` and `{{var}}`) from prompt content.
pub fn extract_template_vars(content: &str) -> Vec<String> {
    let mut vars = Vec::new();
    let bytes = content.as_bytes();

    // Extract %%VAR%% placeholders
    let mut i = 0;
    while i < bytes.len().saturating_sub(3) {
        if bytes[i] == b'%' && bytes[i + 1] == b'%' {
            if let Some(end) = content[i + 2..].find("%%") {
                let var = &content[i..i + 2 + end + 2];
                if !vars.contains(&var.to_string()) {
                    vars.push(var.to_string());
                }
                i += 2 + end + 2;
                continue;
            }
        }
        i += 1;
    }

    // Extract {{var}} placeholders
    let mut i = 0;
    while i < bytes.len().saturating_sub(3) {
        if bytes[i] == b'{' && bytes[i + 1] == b'{' {
            if let Some(end) = content[i + 2..].find("}}") {
                let var = &content[i..i + 2 + end + 2];
                if !vars.contains(&var.to_string()) {
                    vars.push(var.to_string());
                }
                i += 2 + end + 2;
                continue;
            }
        }
        i += 1;
    }

    vars
}

/// Derives a human-readable description from a prompt name and content.
pub fn describe_prompt(name: &str, content: &str) -> String {
    // Try to extract the first markdown heading
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(heading) = trimmed.strip_prefix("# ") {
            return heading.to_string();
        }
    }
    // Fall back to prettifying the name
    name.replace('_', " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_defaults_are_non_empty() {
        let defaults = embedded_defaults();
        for (name, content) in &defaults {
            assert!(!content.is_empty(), "prompt {name} should not be empty");
        }
    }

    #[test]
    fn test_embedded_defaults_count() {
        let defaults = embedded_defaults();
        assert_eq!(defaults.len(), 11);
    }

    #[test]
    fn test_template_variables_preserved_in_embedded() {
        assert!(ORCHESTRATOR_PREAMBLE.contains("{{tools_section}}"));
        assert!(ORCHESTRATOR_PREAMBLE.contains("{{orchestration_system_prompt}}"));
        assert!(WORKER_PREAMBLE.contains("{{worker_system_prompt}}"));
        assert!(WORKER_TASK_PROMPT.contains("%%YOUR_TASK%%"));
        assert!(WORKER_TASK_PROMPT.contains("%%ORCHESTRATION_GOAL%%"));
        assert!(SYNTHESIS_PROMPT.contains("%%GOAL%%"));
        assert!(SYNTHESIS_PROMPT.contains("%%QUERY%%"));
        assert!(SYNTHESIS_PROMPT.contains("%%RESULTS%%"));
        assert!(EVALUATION_PROMPT.contains("%%QUERY%%"));
        assert!(EVALUATION_PROMPT.contains("%%RESULT%%"));
        assert!(REFLECTION_PROMPT.contains("%%ITERATION%%"));
        assert!(REFLECTION_PROMPT.contains("%%GOAL%%"));
        assert!(PHASE_CONTINUATION_PROMPT.contains("%%GOAL%%"));
        assert!(SESSION_HISTORY_TEMPLATE.contains("%%TURN_ENTRIES%%"));
    }

    #[test]
    fn test_extract_template_vars_percent() {
        let content = "Goal: %%GOAL%%\nQuery: %%QUERY%%";
        let vars = extract_template_vars(content);
        assert_eq!(vars, vec!["%%GOAL%%", "%%QUERY%%"]);
    }

    #[test]
    fn test_extract_template_vars_mustache() {
        let content = "Tools: {{tools_section}}\nPrompt: {{system_prompt}}";
        let vars = extract_template_vars(content);
        assert_eq!(vars, vec!["{{tools_section}}", "{{system_prompt}}"]);
    }

    #[test]
    fn test_extract_template_vars_mixed() {
        let content = "%%GOAL%%\n{{tools}}\n%%QUERY%%";
        let vars = extract_template_vars(content);
        assert_eq!(vars, vec!["%%GOAL%%", "%%QUERY%%", "{{tools}}"]);
    }

    #[test]
    fn test_extract_template_vars_no_duplicates() {
        let content = "%%GOAL%% and again %%GOAL%%";
        let vars = extract_template_vars(content);
        assert_eq!(vars, vec!["%%GOAL%%"]);
    }

    #[test]
    fn test_merge_prompt_maps() {
        let mut base = BTreeMap::new();
        base.insert("a".into(), "original_a".into());
        base.insert("b".into(), "original_b".into());

        let mut overrides = BTreeMap::new();
        overrides.insert("b".into(), "overridden_b".into());
        overrides.insert("c".into(), "new_c".into());

        let merged = merge_prompt_maps(&base, &overrides);
        assert_eq!(merged["a"], "original_a");
        assert_eq!(merged["b"], "overridden_b");
        assert_eq!(merged["c"], "new_c");
    }

    #[test]
    fn test_describe_prompt_with_heading() {
        let content = "# Orchestration Coordinator\n\nYou are a coordinator...";
        assert_eq!(describe_prompt("orchestrator_preamble", content), "Orchestration Coordinator");
    }

    #[test]
    fn test_describe_prompt_without_heading() {
        let content = "You are an evaluation agent.";
        assert_eq!(describe_prompt("evaluation_preamble", content), "evaluation preamble");
    }

    #[test]
    fn test_discover_from_dir_nonexistent() {
        let result = discover_from_dir(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }

    #[test]
    fn test_discover_from_dir_reads_embedded_files() {
        // Discover from our own embedded .md files (they exist at src/prompts/)
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/prompts");
        if dir.is_dir() {
            let prompts = discover_from_dir(&dir).unwrap();
            assert!(!prompts.is_empty());
            // Each discovered prompt should have content and extracted vars
            for p in &prompts {
                assert!(!p.content.is_empty(), "{} should have content", p.name);
                assert!(!p.name.is_empty());
            }
        }
    }
}
