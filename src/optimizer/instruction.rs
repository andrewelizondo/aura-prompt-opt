use crate::config::schema::AuraConfig;
use crate::error::Result;
use crate::eval::{EvalDataset, EvalRunner};
use crate::llm::LlmClient;
use crate::llm::client::ChatMessage;
use crate::optimizer::{
    average_score, strip_code_fences, OptimizableField, OptimizationLogEntry,
    OptimizationResult, ScoredCandidate,
};

pub struct InstructionOptimizer {
    pub client: LlmClient,
    pub num_candidates: usize,
    pub target_fields: Vec<OptimizableField>,
}

impl InstructionOptimizer {
    pub fn new(client: LlmClient) -> Self {
        Self {
            client,
            num_candidates: 3,
            target_fields: vec![OptimizableField::AgentSystemPrompt],
        }
    }

    pub fn with_num_candidates(mut self, n: usize) -> Self {
        self.num_candidates = n;
        self
    }

    /// Set specific fields to optimize. If not called, defaults to `agent.system_prompt` only.
    pub fn with_target_fields(mut self, fields: Vec<OptimizableField>) -> Self {
        self.target_fields = fields;
        self
    }

    /// Automatically target all optimizable fields found in the config.
    pub fn with_all_fields(mut self, config: &AuraConfig) -> Self {
        self.target_fields = OptimizableField::all_fields(config);
        self
    }

    pub async fn optimize(
        &self,
        config: &AuraConfig,
        dataset: &EvalDataset,
        runner: &EvalRunner,
    ) -> Result<OptimizationResult> {
        // Evaluate baseline
        let mut baseline_results = Vec::new();
        for scenario in &dataset.scenarios {
            let result = runner.run_scenario(config, scenario).await?;
            baseline_results.push(result);
        }
        let baseline_score = average_score(&baseline_results);

        // Collect failures
        let failures: Vec<_> = baseline_results
            .iter()
            .zip(dataset.scenarios.iter())
            .filter(|(r, _)| !r.passed)
            .collect();

        if failures.is_empty() {
            return Ok(OptimizationResult {
                best_config: config.clone(),
                best_score: baseline_score,
                candidates: vec![ScoredCandidate {
                    config: config.clone(),
                    score: baseline_score,
                    results: baseline_results,
                    notes: vec!["All scenarios passed; no instruction changes needed.".into()],
                }],
                baseline_score,
                optimization_log: vec![],
            });
        }

        // Build failure summary
        let failure_summary = failures
            .iter()
            .take(3)
            .map(|(r, s)| format!("Input: {}\nOutput: {}\nScore: {:.2}", s.input, r.output, r.aggregate_score))
            .collect::<Vec<_>>()
            .join("\n---\n");

        // Optimize each target field
        let mut best_config = config.clone();
        let mut best_score = baseline_score;
        let mut all_scored = vec![ScoredCandidate {
            config: config.clone(),
            score: baseline_score,
            results: baseline_results.clone(),
            notes: vec!["Baseline".into()],
        }];
        let mut log_entries = Vec::new();

        for field in &self.target_fields {
            let current_value = match field.get_value(&best_config) {
                Some(v) => v.to_string(),
                None => continue,
            };

            let prompt = build_optimization_prompt(
                field,
                &current_value,
                &failure_summary,
                self.num_candidates,
            );

            let messages = vec![
                ChatMessage::system(
                    "You are an expert prompt engineer. Return only valid JSON arrays. \
                     When optimizing template prompts, preserve all template variables \
                     (%%VAR%% and {{var}} placeholders) exactly as they appear.",
                ),
                ChatMessage::user(prompt),
            ];

            let response = self.client.chat(&messages).await?;
            let candidates = parse_candidate_prompts(&response)?;

            for candidate_prompt in &candidates {
                // Validate template variables are preserved
                if field.is_template() && !templates_preserved(&current_value, candidate_prompt) {
                    continue; // Skip candidates that drop template variables
                }

                let mut candidate_config = best_config.clone();
                field.set_value(&mut candidate_config, candidate_prompt.clone());

                let mut results = Vec::new();
                for scenario in &dataset.scenarios {
                    let result = runner.run_scenario(&candidate_config, scenario).await?;
                    results.push(result);
                }
                let score = average_score(&results);

                all_scored.push(ScoredCandidate {
                    config: candidate_config,
                    score,
                    results,
                    notes: vec![format!(
                        "InstructionOptimizer candidate for {}",
                        field.field_path()
                    )],
                });
            }

            // Find best candidate for this field
            all_scored.sort_by(|a, b| b.score.total_cmp(&a.score));
            let field_best = &all_scored[0];

            if field_best.score > best_score {
                log_entries.push(OptimizationLogEntry {
                    optimizer: "InstructionOptimizer".into(),
                    field: field.field_path(),
                    before: current_value.clone(),
                    after: field.get_value(&field_best.config)
                        .unwrap_or(&current_value)
                        .to_string(),
                    score_before: best_score,
                    score_after: field_best.score,
                    rationale: format!(
                        "LLM proposed {} candidate prompts for {}; best improved score from {:.2} to {:.2}.",
                        candidates.len(),
                        field.field_path(),
                        best_score,
                        field_best.score,
                    ),
                    alternatives: all_scored.iter().skip(1)
                        .filter_map(|c| {
                            field.get_value(&c.config)
                                .map(|v| (v.to_string(), c.score))
                        })
                        .collect(),
                });

                best_score = field_best.score;
                best_config = field_best.config.clone();
            }
        }

        Ok(OptimizationResult {
            best_config: best_config.clone(),
            best_score,
            candidates: all_scored,
            baseline_score,
            optimization_log: log_entries,
        })
    }
}

/// Builds the meta-prompt for the LLM to generate candidate improvements.
fn build_optimization_prompt(
    field: &OptimizableField,
    current_value: &str,
    failure_summary: &str,
    num_candidates: usize,
) -> String {
    let field_desc = field.description();
    let field_path = field.field_path();

    let template_instruction = if field.is_template() {
        "\n\nCRITICAL: This is a template prompt. You MUST preserve ALL template variables \
         (%%VARIABLE_NAME%% and {{variable_name}} placeholders) exactly as they appear. \
         Do not rename, remove, or add template variables. Only improve the instructional \
         text, structure, and clarity around the existing variables."
    } else {
        ""
    };

    format!(
        "You are a prompt engineer. Analyze these agent failures and propose {num_candidates} \
         improved versions of the prompt field `{field_path}`.\n\n\
         Field description: {field_desc}\n\n\
         Current prompt:\n{current_value}\n\n\
         Failed scenarios:\n{failure_summary}\n\n\
         Return a JSON array of {num_candidates} improved prompts. Format:\n\
         [\"improved prompt 1\", \"improved prompt 2\", ...]{template_instruction}",
    )
}

/// Checks that all %%VAR%% and {{var}} placeholders from the original are in the candidate.
fn templates_preserved(original: &str, candidate: &str) -> bool {
    // Check %%VAR%% placeholders
    let mut i = 0;
    let orig_bytes = original.as_bytes();
    while i < orig_bytes.len().saturating_sub(3) {
        if orig_bytes[i] == b'%' && orig_bytes[i + 1] == b'%' {
            if let Some(end) = original[i + 2..].find("%%") {
                let var = &original[i..i + 2 + end + 2];
                if !candidate.contains(var) {
                    return false;
                }
                i += 2 + end + 2;
                continue;
            }
        }
        i += 1;
    }

    // Check {{var}} placeholders
    let mut i = 0;
    while i < orig_bytes.len().saturating_sub(3) {
        if orig_bytes[i] == b'{' && orig_bytes[i + 1] == b'{' {
            if let Some(end) = original[i + 2..].find("}}") {
                let var = &original[i..i + 2 + end + 2];
                if !candidate.contains(var) {
                    return false;
                }
                i += 2 + end + 2;
                continue;
            }
        }
        i += 1;
    }

    true
}

fn parse_candidate_prompts(response: &str) -> Result<Vec<String>> {
    let s = strip_code_fences(response);
    let v: Vec<String> = serde_json::from_str(s).map_err(|e| {
        crate::error::Error::LlmResponse(format!("failed to parse candidate prompts: {e}"))
    })?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_candidate_prompts_valid() {
        let response = r#"["prompt one", "prompt two", "prompt three"]"#;
        let prompts = parse_candidate_prompts(response).unwrap();
        assert_eq!(prompts.len(), 3);
        assert_eq!(prompts[0], "prompt one");
    }

    #[test]
    fn test_parse_candidate_prompts_with_markdown() {
        let response = "```json\n[\"p1\", \"p2\"]\n```";
        let prompts = parse_candidate_prompts(response).unwrap();
        assert_eq!(prompts.len(), 2);
    }

    #[test]
    fn test_parse_candidate_prompts_invalid() {
        let response = "not json at all";
        assert!(parse_candidate_prompts(response).is_err());
    }

    #[test]
    fn test_templates_preserved_percent_vars() {
        let original = "Goal: %%GOAL%%\nQuery: %%QUERY%%\nResults: %%RESULTS%%";
        let good = "Improved preamble.\nGoal: %%GOAL%%\nQuery: %%QUERY%%\nResults: %%RESULTS%%";
        let bad = "Improved preamble.\nGoal: %%GOAL%%\nQuery: %%QUERY%%";
        assert!(templates_preserved(original, good));
        assert!(!templates_preserved(original, bad));
    }

    #[test]
    fn test_templates_preserved_mustache_vars() {
        let original = "Tools: {{tools_section}}\nPrompt: {{orchestration_system_prompt}}";
        let good = "Better intro.\nTools: {{tools_section}}\nPrompt: {{orchestration_system_prompt}}";
        let bad = "Better intro.\nTools: {{tools_section}}";
        assert!(templates_preserved(original, good));
        assert!(!templates_preserved(original, bad));
    }

    #[test]
    fn test_templates_preserved_no_vars() {
        let original = "Just a plain prompt with no variables.";
        let candidate = "A completely different prompt.";
        assert!(templates_preserved(original, candidate));
    }

    #[test]
    fn test_build_optimization_prompt_includes_template_warning() {
        let field = OptimizableField::OrchestrationPrompt(
            "orchestration.prompts.synthesis_prompt".into(),
        );
        let prompt = build_optimization_prompt(&field, "test %%GOAL%%", "failures", 3);
        assert!(prompt.contains("CRITICAL: This is a template prompt"));
        assert!(prompt.contains("%%VARIABLE_NAME%%"));
    }

    #[test]
    fn test_build_optimization_prompt_no_template_warning_for_agent() {
        let field = OptimizableField::AgentSystemPrompt;
        let prompt = build_optimization_prompt(&field, "test prompt", "failures", 3);
        assert!(!prompt.contains("CRITICAL: This is a template prompt"));
    }
}
