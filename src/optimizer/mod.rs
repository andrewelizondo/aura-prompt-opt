pub mod bootstrap;
pub mod instruction;
pub mod mipro;

pub use bootstrap::BootstrapFewShot;
pub use instruction::InstructionOptimizer;
pub use mipro::MiproOptimizer;

use crate::config::schema::AuraConfig;
use crate::eval::EvalResult;

/// A candidate config with its evaluation score.
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    pub config: AuraConfig,
    pub score: f64,
    pub results: Vec<EvalResult>,
    pub notes: Vec<String>,
}

/// The full output of an optimization run.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub best_config: AuraConfig,
    pub best_score: f64,
    pub candidates: Vec<ScoredCandidate>,
    pub baseline_score: f64,
    pub optimization_log: Vec<OptimizationLogEntry>,
}

/// A single optimization decision recorded for the verbose output.
#[derive(Debug, Clone)]
pub struct OptimizationLogEntry {
    pub optimizer: String,
    pub field: String,
    pub before: String,
    pub after: String,
    pub score_before: f64,
    pub score_after: f64,
    pub rationale: String,
    pub alternatives: Vec<(String, f64)>,
}
