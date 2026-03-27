use crate::config::schema::AuraConfig;
use crate::error::Result;
use crate::eval::{EvalDataset, EvalRunner};
use crate::llm::LlmClient;
use crate::optimizer::{InstructionOptimizer, OptimizationResult, ScoredCandidate};

pub struct MiproOptimizer {
    pub client: LlmClient,
    pub num_rounds: usize,
    pub candidates_per_round: usize,
}

impl MiproOptimizer {
    pub fn new(client: LlmClient) -> Self {
        Self { client, num_rounds: 3, candidates_per_round: 3 }
    }

    pub fn with_rounds(mut self, rounds: usize) -> Self {
        self.num_rounds = rounds;
        self
    }

    pub fn with_candidates_per_round(mut self, n: usize) -> Self {
        self.candidates_per_round = n;
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
                .with_num_candidates(self.candidates_per_round);

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

        all_candidates
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

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

fn average_score(results: &[crate::eval::EvalResult]) -> f64 {
    if results.is_empty() { return 0.0; }
    results.iter().map(|r| r.aggregate_score).sum::<f64>() / results.len() as f64
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
}
