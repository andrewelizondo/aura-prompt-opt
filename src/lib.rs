pub mod compiler;
pub mod config;
pub mod error;
pub mod eval;
pub mod llm;
pub mod optimizer;
pub mod prompts;
pub mod values_file;

pub use error::{Error, Result};

pub use compiler::{compile_optimized, compile_verbose, CompilationOutput};
pub use config::{parse_toml, serialize_toml, AuraConfig};
pub use eval::{
    AgentRunner, AuraHttpRunner, EvalDataset, EvalResult, EvalRunner, EvalScenario,
    ExactMatch, FuzzyMatch, LlmJudge, Metric, OpenAiCompatRunner, StubAgentRunner,
};
pub use llm::LlmClient;
pub use optimizer::{
    BootstrapFewShot, InstructionOptimizer, MiproOptimizer, OptimizableField,
    OptimizationLogEntry, OptimizationResult,
};
pub use prompts::{discover_from_aura_repo, discover_from_dir, DiscoveredPrompt};

use std::path::Path;

pub struct AuraOptimizer {
    pub llm_client: LlmClient,
    pub runner: EvalRunner,
    pub pass_threshold: f64,
    pub max_bootstrap_examples: usize,
    pub instruction_candidates: usize,
}

impl AuraOptimizer {
    pub fn new(llm_client: LlmClient, runner: EvalRunner) -> Self {
        Self {
            llm_client,
            runner,
            pass_threshold: 0.7,
            max_bootstrap_examples: 5,
            instruction_candidates: 3,
        }
    }

    pub fn from_env(metrics: Vec<Box<dyn Metric>>) -> Self {
        let client = LlmClient::from_env();
        let runner = EvalRunner::new(metrics);
        Self::new(client, runner)
    }

    pub fn with_pass_threshold(mut self, threshold: f64) -> Self {
        self.pass_threshold = threshold;
        self
    }

    pub async fn optimize_file(
        &self,
        config_path: &Path,
        dataset: &EvalDataset,
    ) -> Result<CompilationOutput> {
        let toml_str = std::fs::read_to_string(config_path)
            .map_err(|e| Error::Config(format!("failed to read {}: {e}", config_path.display())))?;
        let config = parse_toml(&toml_str)?;
        self.optimize(&config, dataset).await
    }

    pub async fn optimize(
        &self,
        config: &AuraConfig,
        dataset: &EvalDataset,
    ) -> Result<CompilationOutput> {
        // Determine which fields to optimize based on config
        let all_fields = OptimizableField::all_fields(config);
        let has_orchestration = all_fields.iter().any(|f| {
            matches!(f, OptimizableField::OrchestrationPrompt(_) | OptimizableField::WorkerPreamble(_))
        });

        // Stage 1: BootstrapFewShot on the agent system prompt
        let bootstrap = BootstrapFewShot::new()
            .with_pass_threshold(self.pass_threshold)
            .with_max_examples(self.max_bootstrap_examples);
        let bootstrap_result = bootstrap.optimize(config, dataset, &self.runner).await?;

        // Stage 2: InstructionOptimizer — target all fields if orchestration is enabled
        let instruction_opt = if has_orchestration {
            InstructionOptimizer::new(self.llm_client.clone())
                .with_num_candidates(self.instruction_candidates)
                .with_all_fields(config)
        } else {
            InstructionOptimizer::new(self.llm_client.clone())
                .with_num_candidates(self.instruction_candidates)
        };

        let instruction_result = instruction_opt
            .optimize(&bootstrap_result.best_config, dataset, &self.runner)
            .await?;

        let mut all_log = bootstrap_result.optimization_log;
        all_log.extend(instruction_result.optimization_log);

        let final_config = instruction_result.best_config;
        let final_score = instruction_result.best_score;
        let baseline_score = bootstrap_result.baseline_score;

        let optimized_toml = compile_optimized(&final_config)?;
        let verbose_toml = compile_verbose(&final_config, &all_log, baseline_score, final_score)?;

        Ok(CompilationOutput { optimized_toml, verbose_toml })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_public_api_re_exports() {
        let _ = std::mem::size_of::<AuraConfig>();
        let _ = std::mem::size_of::<EvalDataset>();
        let _ = std::mem::size_of::<EvalScenario>();
        let _ = std::mem::size_of::<OptimizationResult>();
        let _ = std::mem::size_of::<OptimizableField>();
    }
}
