use crate::config::schema::AuraConfig;
use crate::error::Result;
use crate::optimizer::OptimizationLogEntry;

pub fn compile_verbose(
    config: &AuraConfig,
    log: &[OptimizationLogEntry],
    baseline_score: f64,
    best_score: f64,
) -> Result<String> {
    let base_toml = toml::to_string_pretty(config)?;
    let mut output = String::new();

    output.push_str(&format!(
        "# ============================================================\n\
         # aura-prompt-opt — Verbose Optimization Report\n\
         # ============================================================\n\
         # Baseline score:  {baseline_score:.3}\n\
         # Optimized score: {best_score:.3}\n\
         # Score delta:     {delta:+.3}\n\
         # Optimizations:   {count}\n\
         # ============================================================\n\n",
        baseline_score = baseline_score,
        best_score = best_score,
        delta = best_score - baseline_score,
        count = log.len(),
    ));

    output.push_str(&annotate_toml(&base_toml, log));
    Ok(output)
}

fn annotate_toml(base_toml: &str, log: &[OptimizationLogEntry]) -> String {
    let mut result = String::new();

    for line in base_toml.lines() {
        let trimmed = line.trim();

        for entry in log {
            if should_annotate(trimmed, &entry.field) {
                result.push_str(&format_annotation(entry));
            }
        }

        result.push_str(line);
        result.push('\n');
    }

    result
}

fn should_annotate(toml_line: &str, field: &str) -> bool {
    match field {
        "agent.system_prompt" => toml_line == "[agent]" || toml_line.starts_with("system_prompt"),
        "agent.turn_depth" => toml_line.starts_with("turn_depth"),
        "agent.temperature" => toml_line.starts_with("temperature"),
        _ if field.starts_with("orchestration.prompts.") => {
            // Match the TOML key for any orchestration prompt field
            let key = field.rsplit('.').next().unwrap_or("");
            toml_line == "[orchestration.prompts]"
                || toml_line == "[orchestration]"
                || toml_line.starts_with(key)
        }
        _ if field.starts_with("orchestration.workers.") => {
            // Match worker preamble fields like orchestration.workers.foo.preamble
            let parts: Vec<&str> = field.split('.').collect();
            if parts.len() >= 4 {
                let worker_name = parts[2];
                toml_line.contains(worker_name) && toml_line.contains("preamble")
                    || toml_line == format!("[orchestration.workers.{worker_name}]")
            } else {
                false
            }
        }
        _ => false,
    }
}

fn format_annotation(entry: &OptimizationLogEntry) -> String {
    let confidence = confidence_label(entry.score_before, entry.score_after);
    let mut s = String::new();

    s.push_str(&format!(
        "# OPTIMIZED by {optimizer}: {field}\n\
         # Before score: {before:.3} → After score: {after:.3} ({delta:+.3})\n\
         # Confidence: {confidence}\n\
         # Rationale: {rationale}\n",
        optimizer = entry.optimizer,
        field = entry.field,
        before = entry.score_before,
        after = entry.score_after,
        delta = entry.score_after - entry.score_before,
        confidence = confidence,
        rationale = entry.rationale,
    ));

    if !entry.alternatives.is_empty() {
        s.push_str("# Alternatives tried:\n");
        for (alt, score) in entry.alternatives.iter().take(5) {
            let preview: String = alt.chars().take(60).collect();
            s.push_str(&format!("#   score={score:.3}: \"{preview}...\"\n"));
        }
    }

    let before_preview: String = entry.before.chars().take(100).collect();
    let after_preview: String = entry.after.chars().take(100).collect();
    if before_preview != after_preview {
        s.push_str(&format!(
            "# Before: \"{before_preview}...\"\n\
             # After:  \"{after_preview}...\"\n",
        ));
    }

    s.push_str("#\n");
    s
}

fn confidence_label(score_before: f64, score_after: f64) -> &'static str {
    let delta = score_after - score_before;
    if delta >= 0.3 { "HIGH" }
    else if delta >= 0.1 { "MEDIUM" }
    else if delta > 0.0 { "LOW" }
    else { "NEUTRAL (no improvement)" }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::{AgentConfig, AuraConfig, LlmConfig};
    use crate::optimizer::OptimizationLogEntry;

    fn make_config() -> AuraConfig {
        AuraConfig {
            llm: LlmConfig::OpenAI { api_key: "sk-test".into(), model: "gpt-4o".into(), base_url: None },
            agent: AgentConfig {
                name: "SRE".into(),
                system_prompt: "You are an SRE.".into(),
                ..AgentConfig::default()
            },
            ..AuraConfig::default()
        }
    }

    #[test]
    fn test_compile_verbose_contains_header() {
        let config = make_config();
        let result = compile_verbose(&config, &[], 0.5, 0.85).unwrap();
        assert!(result.contains("aura-prompt-opt"));
        assert!(result.contains("Baseline score:  0.500"));
        assert!(result.contains("Optimized score: 0.850"));
    }

    #[test]
    fn test_compile_verbose_with_log_entry() {
        let config = make_config();
        let entry = OptimizationLogEntry {
            optimizer: "BootstrapFewShot".into(),
            field: "agent.system_prompt".into(),
            before: "You are an SRE.".into(),
            after: "You are a senior SRE with 10 years experience.".into(),
            score_before: 0.45,
            score_after: 0.82,
            rationale: "Added few-shot examples".into(),
            alternatives: vec![
                ("alt prompt one".into(), 0.67),
                ("alt prompt two".into(), 0.72),
            ],
        };

        let result = compile_verbose(&config, &[entry], 0.45, 0.82).unwrap();
        assert!(result.contains("OPTIMIZED by BootstrapFewShot"));
        assert!(result.contains("Confidence: HIGH"));
        assert!(result.contains("Alternatives tried:"));
        assert!(result.contains("score=0.670"));
    }

    #[test]
    fn test_confidence_labels() {
        assert_eq!(confidence_label(0.4, 0.75), "HIGH");
        assert_eq!(confidence_label(0.6, 0.75), "MEDIUM");
        assert_eq!(confidence_label(0.6, 0.65), "LOW");
        assert_eq!(confidence_label(0.7, 0.65), "NEUTRAL (no improvement)");
    }

    #[test]
    fn test_compile_verbose_is_valid_toml_after_stripping_comments() {
        let config = make_config();
        let result = compile_verbose(&config, &[], 0.5, 0.8).unwrap();
        let toml_only: String = result
            .lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let reparsed: AuraConfig = toml::from_str(&toml_only).unwrap();
        assert_eq!(reparsed.agent.name, "SRE");
    }
}
