use crate::config::schema::AuraConfig;
use crate::error::Result;
use crate::eval::{EvalDataset, EvalRunner};
use crate::llm::LlmClient;
use crate::optimizer::{
    average_score, InstructionOptimizer, OptimizableField, OptimizationResult, ScoredCandidate,
};

pub struct MiproOptimizer {
    pub client: LlmClient,
    pub num_rounds: usize,
    pub candidates_per_round: usize,
    pub target_fields: Vec<OptimizableField>,
}

impl MiproOptimizer {
    pub fn new(client: LlmClient) -> Self {
        Self {
            client,
            num_rounds: 3,
            candidates_per_round: 3,
            target_fields: vec![OptimizableField::AgentSystemPrompt],
        }
    }

    pub fn with_rounds(mut self, rounds: usize) -> Self {
        self.num_rounds = rounds;
        self
    }

    pub fn with_candidates_per_round(mut self, n: usize) -> Self {
        self.candidates_per_round = n;
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
        let mut current_config = config.clone();
        let mut all_candidates: Vec<ScoredCandidate> = Vec::new();
        let mut all_log_entries = Vec::new();

        // Evaluate initial baseline
        let mut baseline_results = Vec::new();
        for scenario in &dataset.scenarios {
            let result = runner.run_scenario(&current_config, scenario).await?;
            baseline_results.push(result);
        }
        let baseline_score = average_score(&baseline_results);
        all_candidates.push(ScoredCandidate {
            config: current_config.clone(),
            score: baseline_score,
            results: baseline_results,
            notes: vec!["Initial baseline (MIPRO round 0)".into()],
        });

        let mut best_score = baseline_score;

        for round in 1..=self.num_rounds {
            let instruction_opt = InstructionOptimizer::new(self.client.clone())
                .with_num_candidates(self.candidates_per_round)
                .with_target_fields(self.target_fields.clone());

            let round_result = instruction_opt
                .optimize(&current_config, dataset, runner)
                .await?;

            for mut candidate in round_result.candidates {
                candidate.notes.push(format!("MIPRO round {round}"));
                all_candidates.push(candidate);
            }

            for mut entry in round_result.optimization_log {
                entry.optimizer = format!("MIPRO(round={round})/InstructionOptimizer");
                all_log_entries.push(entry);
            }

            if round_result.best_score > best_score {
                best_score = round_result.best_score;
                current_config = round_result.best_config;
            }
        }

        all_candidates.sort_by(|a, b| b.score.total_cmp(&a.score));

        let best_candidate = all_candidates.first().unwrap().clone();

        Ok(OptimizationResult {
            best_config: best_candidate.config.clone(),
            best_score: best_candidate.score,
            candidates: all_candidates,
            baseline_score,
            optimization_log: all_log_entries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LlmClient;

    #[test]
    fn test_mipro_builder() {
        let client = LlmClient::new("http://localhost", "key", "gpt-4o");
        let opt = MiproOptimizer::new(client).with_rounds(2).with_candidates_per_round(4);
        assert_eq!(opt.num_rounds, 2);
        assert_eq!(opt.candidates_per_round, 4);
    }

    #[test]
    fn test_mipro_target_fields_default() {
        let client = LlmClient::new("http://localhost", "key", "gpt-4o");
        let opt = MiproOptimizer::new(client);
        assert_eq!(opt.target_fields.len(), 1);
        assert_eq!(opt.target_fields[0], OptimizableField::AgentSystemPrompt);
    }
}
