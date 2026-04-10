use crate::config::schema::AuraConfig;
use crate::error::Result;
use crate::eval::{EvalDataset, EvalResult, EvalRunner};
use crate::optimizer::{
    average_score, OptimizableField, OptimizationLogEntry, OptimizationResult, ScoredCandidate,
};

pub struct BootstrapFewShot {
    pub pass_threshold: f64,
    pub max_examples: usize,
    pub target_field: OptimizableField,
}

impl BootstrapFewShot {
    pub fn new() -> Self {
        Self {
            pass_threshold: 0.7,
            max_examples: 5,
            target_field: OptimizableField::AgentSystemPrompt,
        }
    }

    pub fn with_pass_threshold(mut self, threshold: f64) -> Self {
        self.pass_threshold = threshold;
        self
    }

    pub fn with_max_examples(mut self, n: usize) -> Self {
        self.max_examples = n;
        self
    }

    /// Set the prompt field to inject few-shot examples into. Defaults to `agent.system_prompt`.
    pub fn with_target_field(mut self, field: OptimizableField) -> Self {
        self.target_field = field;
        self
    }

    pub async fn optimize(
        &self,
        config: &AuraConfig,
        dataset: &EvalDataset,
        runner: &EvalRunner,
    ) -> Result<OptimizationResult> {
        // Step 1: Evaluate baseline
        let mut baseline_results = Vec::new();
        for scenario in &dataset.scenarios {
            let result = runner.run_scenario(config, scenario).await?;
            baseline_results.push(result);
        }
        let baseline_score = average_score(&baseline_results);

        // Step 2: Collect passing examples
        let passing: Vec<(&EvalResult, &crate::eval::EvalScenario)> = baseline_results
            .iter()
            .zip(dataset.scenarios.iter())
            .filter(|(r, _)| r.aggregate_score >= self.pass_threshold)
            .take(self.max_examples)
            .collect();

        if passing.is_empty() {
            let candidate = ScoredCandidate {
                config: config.clone(),
                score: baseline_score,
                results: baseline_results,
                notes: vec!["No passing examples found; returning baseline config.".into()],
            };
            return Ok(OptimizationResult {
                best_config: config.clone(),
                best_score: baseline_score,
                candidates: vec![candidate],
                baseline_score,
                optimization_log: vec![],
            });
        }

        // Step 3: Build few-shot block and inject into the target prompt field
        let few_shot_block = build_few_shot_block(&passing);
        let mut optimized_config = config.clone();

        let original_prompt = self.target_field.get_value(config)
            .unwrap_or("")
            .to_string();
        let new_prompt = format!("{}\n\n{}", original_prompt.trim_end(), few_shot_block);
        self.target_field.set_value(&mut optimized_config, new_prompt.clone());

        // Step 4: Evaluate optimized config
        let mut optimized_results = Vec::new();
        for scenario in &dataset.scenarios {
            let result = runner.run_scenario(&optimized_config, scenario).await?;
            optimized_results.push(result);
        }
        let optimized_score = average_score(&optimized_results);

        let log_entry = OptimizationLogEntry {
            optimizer: "BootstrapFewShot".into(),
            field: self.target_field.field_path(),
            before: original_prompt.clone(),
            after: new_prompt,
            score_before: baseline_score,
            score_after: optimized_score,
            rationale: format!(
                "Injected {} few-shot examples (score >= {:.2}) from baseline evaluation.",
                passing.len(),
                self.pass_threshold
            ),
            alternatives: vec![],
        };

        let best_config = if optimized_score >= baseline_score {
            optimized_config.clone()
        } else {
            config.clone()
        };
        let best_score = f64::max(optimized_score, baseline_score);

        let candidates = vec![
            ScoredCandidate {
                config: config.clone(),
                score: baseline_score,
                results: baseline_results,
                notes: vec!["Baseline config".into()],
            },
            ScoredCandidate {
                config: optimized_config,
                score: optimized_score,
                results: optimized_results,
                notes: vec!["After BootstrapFewShot injection".into()],
            },
        ];

        Ok(OptimizationResult {
            best_config,
            best_score,
            candidates,
            baseline_score,
            optimization_log: vec![log_entry],
        })
    }
}

impl Default for BootstrapFewShot {
    fn default() -> Self { Self::new() }
}

fn build_few_shot_block(
    examples: &[(&EvalResult, &crate::eval::EvalScenario)],
) -> String {
    let mut block = String::from("## Examples\n\n");
    for (result, scenario) in examples {
        block.push_str(&format!("User: {}\nAssistant: {}\n\n", scenario.input, result.output));
    }
    block.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::{AgentConfig, AuraConfig};
    use crate::eval::{EvalDataset, EvalScenario, EvalRunner, ExactMatch};

    fn make_config(prompt: &str) -> AuraConfig {
        AuraConfig {
            agent: AgentConfig { name: "Test".into(), system_prompt: prompt.into(), ..AgentConfig::default() },
            ..AuraConfig::default()
        }
    }

    #[tokio::test]
    async fn test_bootstrap_no_passing_examples() {
        let config = make_config("You are an assistant.");
        let dataset = EvalDataset::new("test")
            .add(EvalScenario::new("s1", "hello").with_expected("exact_match_impossible_xyz"));

        let metric: Box<dyn crate::eval::Metric> = Box::new(ExactMatch::new());
        let runner = EvalRunner::new(vec![metric]);
        let bfs = BootstrapFewShot::new().with_pass_threshold(1.0);

        let result = bfs.optimize(&config, &dataset, &runner).await.unwrap();
        assert_eq!(result.best_config.agent.system_prompt, "You are an assistant.");
        assert_eq!(result.optimization_log.len(), 0);
    }

    #[tokio::test]
    async fn test_bootstrap_with_passing_examples() {
        let config = make_config("You are an assistant.");
        let stub_output = "[STUB] Agent 'Test' received: hello";
        let dataset = EvalDataset::new("test")
            .add(EvalScenario::new("s1", "hello").with_expected(stub_output));

        let metric: Box<dyn crate::eval::Metric> = Box::new(ExactMatch::new());
        let runner = EvalRunner::new(vec![metric]).with_pass_threshold(1.0);
        let bfs = BootstrapFewShot::new().with_pass_threshold(1.0).with_max_examples(3);

        let result = bfs.optimize(&config, &dataset, &runner).await.unwrap();
        assert_eq!(result.optimization_log.len(), 1);
        assert!(result.optimization_log[0].after.contains("## Examples"));
    }

    #[test]
    fn test_bootstrap_target_field_default() {
        let bfs = BootstrapFewShot::new();
        assert_eq!(bfs.target_field, OptimizableField::AgentSystemPrompt);
    }

    #[test]
    fn test_bootstrap_target_field_override() {
        let bfs = BootstrapFewShot::new()
            .with_target_field(OptimizableField::OrchestrationPrompt(
                "orchestration.prompts.orchestrator_preamble".into(),
            ));
        assert_eq!(
            bfs.target_field.field_path(),
            "orchestration.prompts.orchestrator_preamble"
        );
    }
}
