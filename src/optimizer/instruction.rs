use crate::config::schema::AuraConfig;
use crate::error::Result;
use crate::eval::{EvalDataset, EvalRunner};
use crate::llm::LlmClient;
use crate::llm::client::ChatMessage;
use crate::optimizer::{OptimizationLogEntry, OptimizationResult, ScoredCandidate};

pub struct InstructionOptimizer {
    pub client: LlmClient,
    pub num_candidates: usize,
}

impl InstructionOptimizer {
    pub fn new(client: LlmClient) -> Self {
        Self { client, num_candidates: 3 }
    }

    pub fn with_num_candidates(mut self, n: usize) -> Self {
        self.num_candidates = n;
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

        let prompt = format!(
            "You are a prompt engineer. Analyze these agent failures and propose {} improved system prompts.\n\n\
             Current system prompt:\n{}\n\n\
             Failed scenarios:\n{}\n\n\
             Return a JSON array of {} improved system prompts. Format:\n\
             [\"improved prompt 1\", \"improved prompt 2\", ...]",
            self.num_candidates,
            config.agent.system_prompt,
            failure_summary,
            self.num_candidates,
        );

        let messages = vec![
            ChatMessage::system("You are an expert prompt engineer. Return only valid JSON arrays."),
            ChatMessage::user(prompt),
        ];

        let response = self.client.chat(&messages).await?;
        let candidates = parse_candidate_prompts(&response)?;

        // Score each candidate
        let mut scored = vec![ScoredCandidate {
            config: config.clone(),
            score: baseline_score,
            results: baseline_results.clone(),
            notes: vec!["Baseline".into()],
        }];

        let mut log_entries = Vec::new();

        for candidate_prompt in &candidates {
            let mut candidate_config = config.clone();
            candidate_config.agent.system_prompt = candidate_prompt.clone();

            let mut results = Vec::new();
            for scenario in &dataset.scenarios {
                let result = runner.run_scenario(&candidate_config, scenario).await?;
                results.push(result);
            }
            let score = average_score(&results);

            scored.push(ScoredCandidate {
                config: candidate_config,
                score,
                results,
                notes: vec!["InstructionOptimizer candidate".into()],
            });
        }

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let best = scored.first().unwrap().clone();

        if best.score > baseline_score {
            log_entries.push(OptimizationLogEntry {
                optimizer: "InstructionOptimizer".into(),
                field: "agent.system_prompt".into(),
                before: config.agent.system_prompt.clone(),
                after: best.config.agent.system_prompt.clone(),
                score_before: baseline_score,
                score_after: best.score,
                rationale: format!(
                    "LLM proposed {} candidate prompts; best improved score from {:.2} to {:.2}.",
                    candidates.len(), baseline_score, best.score
                ),
                alternatives: scored.iter().skip(1)
                    .map(|c| (c.config.agent.system_prompt.clone(), c.score))
                    .collect(),
            });
        }

        Ok(OptimizationResult {
            best_config: best.config.clone(),
            best_score: best.score,
            candidates: scored,
            baseline_score,
            optimization_log: log_entries,
        })
    }
}

fn average_score(results: &[crate::eval::EvalResult]) -> f64 {
    if results.is_empty() { return 0.0; }
    results.iter().map(|r| r.aggregate_score).sum::<f64>() / results.len() as f64
}

fn parse_candidate_prompts(response: &str) -> Result<Vec<String>> {
    let s = response.trim();
    let s = s.strip_prefix("```json").unwrap_or(s);
    let s = s.strip_prefix("```").unwrap_or(s);
    let s = s.strip_suffix("```").unwrap_or(s);
    let s = s.trim();

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
}
