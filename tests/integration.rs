use aura_prompt_opt::{
    compiler::{compile_optimized, compile_verbose},
    config::parse_toml,
    eval::{EvalDataset, EvalRunner, EvalScenario, ExactMatch, FuzzyMatch},
    optimizer::{BootstrapFewShot, OptimizableField, OptimizationLogEntry},
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

// ── Orchestration mode integration tests ───────────────────────────────

const ORCHESTRATION_TOML: &str = r#"
[llm]
provider = "anthropic"
api_key = "sk-test"
model = "claude-sonnet-4-6"

[agent]
name = "Platform Agent"
system_prompt = "You are a platform engineering agent."
turn_depth = 20

[orchestration]
enabled = true
max_planning_cycles = 3
quality_threshold = 0.8

[orchestration.workers.database]
description = "Executes SQL queries"
preamble = "You are a database specialist."
mcp_filter = ["database"]

[orchestration.workers.monitoring]
description = "Queries metrics and logs"
preamble = "You are a monitoring specialist."
mcp_filter = ["monitoring"]
"#;

#[test]
fn test_parse_orchestration_config() {
    let config = parse_toml(ORCHESTRATION_TOML).unwrap();
    let orch = config.orchestration.as_ref().unwrap();
    assert!(orch.enabled);
    assert_eq!(orch.max_planning_cycles, 3);
    assert_eq!(orch.workers.len(), 2);
    assert!(orch.workers.contains_key("database"));
    assert!(orch.workers.contains_key("monitoring"));
    // Prompts should have defaults
    assert!(!orch.prompts.orchestrator_preamble.is_empty());
    assert!(orch.prompts.orchestrator_preamble.contains("Orchestration Coordinator"));
    assert!(orch.prompts.synthesis_prompt.contains("%%GOAL%%"));
}

#[test]
fn test_orchestration_prompts_round_trip() {
    let config = parse_toml(ORCHESTRATION_TOML).unwrap();
    let toml_str = compile_optimized(&config).unwrap();
    let reparsed = parse_toml(&toml_str).unwrap();

    let orig = &config.orchestration.as_ref().unwrap().prompts;
    let round = &reparsed.orchestration.as_ref().unwrap().prompts;

    assert_eq!(orig.orchestrator_preamble, round.orchestrator_preamble);
    assert_eq!(orig.worker_preamble, round.worker_preamble);
    assert_eq!(orig.synthesis_prompt, round.synthesis_prompt);
    assert_eq!(orig.evaluation_prompt, round.evaluation_prompt);
    assert_eq!(orig.reflection_prompt, round.reflection_prompt);
    assert_eq!(orig.phase_continuation_prompt, round.phase_continuation_prompt);
}

#[test]
fn test_optimizable_field_all_fields_with_orchestration() {
    let config = parse_toml(ORCHESTRATION_TOML).unwrap();
    let fields = OptimizableField::all_fields(&config);

    // Should include: agent.system_prompt + 11 orchestration prompts + 2 worker preambles = 14
    assert_eq!(fields.len(), 14);

    // Check that agent system prompt is included
    assert!(fields.iter().any(|f| matches!(f, OptimizableField::AgentSystemPrompt)));

    // Check orchestration prompts
    assert!(fields.iter().any(|f| {
        matches!(f, OptimizableField::OrchestrationPrompt(p) if p == "orchestration.prompts.orchestrator_preamble")
    }));
    assert!(fields.iter().any(|f| {
        matches!(f, OptimizableField::OrchestrationPrompt(p) if p == "orchestration.prompts.synthesis_prompt")
    }));
    assert!(fields.iter().any(|f| {
        matches!(f, OptimizableField::OrchestrationPrompt(p) if p == "orchestration.prompts.evaluation_prompt")
    }));

    // Check worker preambles
    assert!(fields.iter().any(|f| {
        matches!(f, OptimizableField::WorkerPreamble(n) if n == "database")
    }));
    assert!(fields.iter().any(|f| {
        matches!(f, OptimizableField::WorkerPreamble(n) if n == "monitoring")
    }));
}

#[test]
fn test_optimizable_field_get_set_orchestration_prompt() {
    let mut config = parse_toml(ORCHESTRATION_TOML).unwrap();

    let field = OptimizableField::OrchestrationPrompt(
        "orchestration.prompts.synthesis_prompt".to_string(),
    );

    // Get default value
    let original = field.get_value(&config).unwrap().to_string();
    assert!(original.contains("%%GOAL%%"));

    // Set new value
    let new_value = "Improved synthesis prompt with %%GOAL%% and %%QUERY%%".to_string();
    assert!(field.set_value(&mut config, new_value.clone()));

    // Verify it was set
    let updated = field.get_value(&config).unwrap();
    assert_eq!(updated, new_value);
}

#[test]
fn test_optimizable_field_get_set_worker_preamble() {
    let mut config = parse_toml(ORCHESTRATION_TOML).unwrap();

    let field = OptimizableField::WorkerPreamble("database".to_string());

    // Get original
    let original = field.get_value(&config).unwrap();
    assert_eq!(original, "You are a database specialist.");

    // Set new value
    let new_value = "You are an expert database administrator with deep PostgreSQL knowledge.".to_string();
    assert!(field.set_value(&mut config, new_value.clone()));

    // Verify
    let updated = field.get_value(&config).unwrap();
    assert_eq!(updated, new_value);
}

#[test]
fn test_all_fields_without_orchestration() {
    let config = parse_toml(SRE_TOML).unwrap();
    let fields = OptimizableField::all_fields(&config);

    // Without orchestration enabled, only agent.system_prompt
    assert_eq!(fields.len(), 1);
    assert!(matches!(fields[0], OptimizableField::AgentSystemPrompt));
}

#[test]
fn test_verbose_compiler_annotates_orchestration_prompts() {
    let config = parse_toml(ORCHESTRATION_TOML).unwrap();
    let log = vec![OptimizationLogEntry {
        optimizer: "InstructionOptimizer".into(),
        field: "orchestration.prompts.synthesis_prompt".into(),
        before: "old synthesis".into(),
        after: "new synthesis".into(),
        score_before: 0.5,
        score_after: 0.8,
        rationale: "Improved synthesis clarity.".into(),
        alternatives: vec![],
    }];

    let verbose = compile_verbose(&config, &log, 0.5, 0.8).unwrap();
    assert!(verbose.contains("OPTIMIZED by InstructionOptimizer"));
    assert!(verbose.contains("orchestration.prompts.synthesis_prompt"));
}

#[test]
fn test_orchestration_prompts_contain_expected_template_vars() {
    let config = parse_toml(ORCHESTRATION_TOML).unwrap();
    let prompts = &config.orchestration.as_ref().unwrap().prompts;

    // Verify key template variables are present in the defaults
    assert!(prompts.orchestrator_preamble.contains("{{tools_section}}"));
    assert!(prompts.orchestrator_preamble.contains("{{orchestration_system_prompt}}"));
    assert!(prompts.worker_preamble.contains("{{worker_system_prompt}}"));
    assert!(prompts.worker_task_prompt.contains("%%YOUR_TASK%%"));
    assert!(prompts.synthesis_prompt.contains("%%GOAL%%"));
    assert!(prompts.synthesis_prompt.contains("%%QUERY%%"));
    assert!(prompts.synthesis_prompt.contains("%%RESULTS%%"));
    assert!(prompts.evaluation_prompt.contains("%%QUERY%%"));
    assert!(prompts.evaluation_prompt.contains("%%RESULT%%"));
    assert!(prompts.reflection_prompt.contains("%%ITERATION%%"));
    assert!(prompts.phase_continuation_prompt.contains("%%GOAL%%"));
    assert!(prompts.session_history_template.contains("%%TURN_ENTRIES%%"));
}

#[tokio::test]
async fn test_bootstrap_with_orchestration_field() {
    let config = parse_toml(ORCHESTRATION_TOML).unwrap();
    let stub_output = "[STUB] Agent 'Platform Agent' received: hello";
    let dataset = EvalDataset::new("test")
        .add(EvalScenario::new("s1", "hello").with_expected(stub_output));

    let metric: Box<dyn aura_prompt_opt::Metric> = Box::new(ExactMatch::new());
    let runner = EvalRunner::new(vec![metric]).with_pass_threshold(1.0);

    // Target the orchestrator preamble instead of agent system prompt
    let bfs = BootstrapFewShot::new()
        .with_pass_threshold(1.0)
        .with_max_examples(3)
        .with_target_field(OptimizableField::OrchestrationPrompt(
            "orchestration.prompts.orchestrator_preamble".into(),
        ));

    let result = bfs.optimize(&config, &dataset, &runner).await.unwrap();
    assert_eq!(result.optimization_log.len(), 1);
    assert_eq!(
        result.optimization_log[0].field,
        "orchestration.prompts.orchestrator_preamble"
    );
    assert!(result.optimization_log[0].after.contains("## Examples"));
}
