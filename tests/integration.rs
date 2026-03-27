use aura_prompt_opt::{
    compiler::{compile_optimized, compile_verbose},
    config::parse_toml,
    eval::{EvalDataset, EvalRunner, EvalScenario, ExactMatch, FuzzyMatch},
    optimizer::{BootstrapFewShot, OptimizationLogEntry},
};

const SRE_TOML: &str = r#"
[llm]
provider = "openai"
api_key = "sk-test"
model = "gpt-4o"

[mcp]
sanitize_schemas = true

[mcp.servers.kubernetes]
transport = "http_streamable"
url = "http://localhost:8081/mcp"
description = "Kubernetes cluster operations"

[agent]
name = "SRE Agent"
system_prompt = "You are an SRE agent. Help with incidents."
turn_depth = 20
"#;

#[test]
fn test_parse_sre_config() {
    let config = parse_toml(SRE_TOML).unwrap();
    assert_eq!(config.agent.name, "SRE Agent");
    assert_eq!(config.agent.turn_depth, Some(20));
    let mcp = config.mcp.as_ref().unwrap();
    assert!(mcp.servers.contains_key("kubernetes"));
}

#[test]
fn test_compile_optimized_round_trip() {
    let config = parse_toml(SRE_TOML).unwrap();
    let toml_str = compile_optimized(&config).unwrap();
    let reparsed = parse_toml(&toml_str).unwrap();
    assert_eq!(config.agent.name, reparsed.agent.name);
    assert_eq!(config.agent.system_prompt, reparsed.agent.system_prompt);
}

#[test]
fn test_compile_verbose_has_header_and_is_valid() {
    let config = parse_toml(SRE_TOML).unwrap();
    let log = vec![OptimizationLogEntry {
        optimizer: "BootstrapFewShot".into(),
        field: "agent.system_prompt".into(),
        before: "You are an SRE agent. Help with incidents.".into(),
        after: "You are a senior SRE. Diagnose incidents systematically.".into(),
        score_before: 0.45,
        score_after: 0.82,
        rationale: "Few-shot examples improved clarity.".into(),
        alternatives: vec![("alt prompt".into(), 0.65)],
    }];

    let verbose = compile_verbose(&config, &log, 0.45, 0.82).unwrap();
    assert!(verbose.contains("aura-prompt-opt"));
    assert!(verbose.contains("OPTIMIZED by BootstrapFewShot"));
    assert!(verbose.contains("Confidence: HIGH"));
    assert!(verbose.contains("[agent]"));
}

#[tokio::test]
async fn test_bootstrap_no_examples_returns_baseline() {
    let config = parse_toml(SRE_TOML).unwrap();
    let dataset = EvalDataset::new("test").add(
        EvalScenario::new("s1", "what happened?").with_expected("impossible_exact_match_xyz"),
    );

    let metric: Box<dyn aura_prompt_opt::Metric> = Box::new(ExactMatch::new());
    let runner = EvalRunner::new(vec![metric]);
    let bfs = BootstrapFewShot::new().with_pass_threshold(1.0);

    let result = bfs.optimize(&config, &dataset, &runner).await.unwrap();
    assert_eq!(result.best_config.agent.system_prompt, config.agent.system_prompt);
}

#[tokio::test]
async fn test_eval_runner_stub_returns_output() {
    let config = parse_toml(SRE_TOML).unwrap();
    let runner = EvalRunner::new(vec![]);
    let output = runner.run_agent(&config, "test input").await.unwrap();
    assert!(output.contains("SRE Agent"));
    assert!(output.contains("test input"));
}

#[tokio::test]
async fn test_fuzzy_metric_in_eval_runner() {
    let config = parse_toml(SRE_TOML).unwrap();
    let scenario = EvalScenario::new("test", "hello").with_expected("SRE Agent");

    let metric: Box<dyn aura_prompt_opt::Metric> = Box::new(FuzzyMatch::new());
    let runner = EvalRunner::new(vec![metric]);
    let result = runner.run_scenario(&config, &scenario).await.unwrap();
    assert!(result.passed);
    assert_eq!(result.aggregate_score, 1.0);
}
