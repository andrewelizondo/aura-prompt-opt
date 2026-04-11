use crate::config::schema::AuraConfig;
use crate::error::Result;
use crate::eval::agent_runner::{AgentRunner, StubAgentRunner};
use crate::eval::metrics::{Metric, MetricScore};
use crate::eval::scenario::EvalScenario;

#[derive(Debug, Clone)]
pub struct EvalResult {
    pub scenario_name: String,
    pub output: String,
    pub scores: Vec<(String, MetricScore)>,
    pub aggregate_score: f64,
    pub passed: bool,
}

pub struct EvalRunner {
    pub metrics: Vec<Box<dyn Metric>>,
    pub pass_threshold: f64,
    pub agent: Box<dyn AgentRunner>,
}

impl EvalRunner {
    /// Creates a new runner with a stub agent (echoes canned responses).
    ///
    /// Use `with_agent()` to inject a real runner for actual optimization.
    pub fn new(metrics: Vec<Box<dyn Metric>>) -> Self {
        Self {
            metrics,
            pass_threshold: 0.7,
            agent: Box::new(StubAgentRunner::new()),
        }
    }

    pub fn with_pass_threshold(mut self, threshold: f64) -> Self {
        self.pass_threshold = threshold;
        self
    }

    /// Injects a concrete `AgentRunner` implementation.
    pub fn with_agent(mut self, agent: Box<dyn AgentRunner>) -> Self {
        self.agent = agent;
        self
    }

    pub async fn run_scenario(
        &self,
        config: &AuraConfig,
        scenario: &EvalScenario,
    ) -> Result<EvalResult> {
        let output = self.agent.run(config, &scenario.input).await?;
        self.score_output(scenario, &output).await
    }

    pub async fn score_output(
        &self,
        scenario: &EvalScenario,
        output: &str,
    ) -> Result<EvalResult> {
        let mut scores = Vec::new();
        let mut total = 0.0;

        for metric in &self.metrics {
            let score = metric.score(scenario, output).await?;
            total += score.score;
            scores.push((metric.name().to_string(), score));
        }

        let aggregate_score = if scores.is_empty() { 0.0 } else { total / scores.len() as f64 };

        Ok(EvalResult {
            scenario_name: scenario.name.clone(),
            output: output.to_string(),
            scores,
            aggregate_score,
            passed: aggregate_score >= self.pass_threshold,
        })
    }

    /// Convenience wrapper that runs the agent directly.
    ///
    /// Retained for backward compatibility. Prefer injecting the runner
    /// via `with_agent()` and calling `run_scenario()`.
    pub async fn run_agent(
        &self,
        config: &AuraConfig,
        input: &str,
    ) -> Result<String> {
        self.agent.run(config, input).await
    }
}
