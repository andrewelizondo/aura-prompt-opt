use aura_prompt_opt::{
    AgentRunner, AuraHttpRunner, AuraOptimizer, EvalDataset, EvalScenario, FuzzyMatch, LlmClient,
    LlmJudge, Metric, OpenAiCompatRunner, StubAgentRunner,
    config::schema::OrchestrationPrompts,
    llm::client::{DEFAULT_BASE_URL, DEFAULT_MODEL},
};
use clap::{Parser, ValueEnum};
use std::path::{Path, PathBuf};

/// aura-prompt-opt: optimize Aura agent TOML configurations.
#[derive(Parser)]
#[command(name = "aura-prompt-opt", version, about)]
struct Cli {
    /// Path to a TOML config file to optimize.
    #[arg(group = "input")]
    config_file: Option<PathBuf>,

    /// Path to a Helm values YAML file containing embedded TOML.
    #[arg(long, group = "input")]
    values_file: Option<PathBuf>,

    /// Dotted key path to the TOML content within the values file.
    #[arg(long, default_value = "config.content")]
    config_key: String,

    /// Path to a local Aura repo for dynamic prompt discovery.
    #[arg(long, env = "AURA_REPO")]
    aura_repo: Option<PathBuf>,

    /// LLM API base URL (OpenAI-compatible) for the optimizer's judge/instructor.
    #[arg(long, env = "OPTIMIZER_BASE_URL", default_value = DEFAULT_BASE_URL)]
    base_url: String,

    /// LLM API key for the optimizer's judge/instructor.
    #[arg(long, env = "OPTIMIZER_API_KEY")]
    api_key: Option<String>,

    /// LLM model name for the optimizer's judge/instructor.
    #[arg(long, env = "OPTIMIZER_MODEL", default_value = DEFAULT_MODEL)]
    model: String,

    /// Which agent runner to use for executing the agent under test.
    #[arg(long, value_enum, default_value_t = AgentKind::Stub)]
    agent: AgentKind,

    /// Base URL for the agent under test. Required when `--agent` is
    /// `aura-http` or `openai-compat`.
    #[arg(long, env = "AGENT_BASE_URL")]
    agent_base_url: Option<String>,

    /// API key for the agent under test (only used by `openai-compat`).
    #[arg(long, env = "AGENT_API_KEY")]
    agent_api_key: Option<String>,

    /// Chat path for the aura-http runner (default: /v1/chat).
    #[arg(long, env = "AURA_CHAT_PATH")]
    aura_chat_path: Option<String>,

    /// Which metric to score agent outputs.
    #[arg(long, value_enum, default_value_t = MetricKind::Fuzzy)]
    metric: MetricKind,

    /// Pass threshold for evaluation (0.0 to 1.0).
    #[arg(long, default_value = "0.6")]
    pass_threshold: f64,

    /// Output file for optimized TOML.
    #[arg(long, short, default_value = "config.optimized.toml")]
    output: PathBuf,

    /// Output file for verbose TOML with optimization annotations.
    #[arg(long, default_value = "config.verbose.toml")]
    verbose_output: PathBuf,

    /// Output path for the updated values YAML (only with --values-file).
    #[arg(long)]
    values_output: Option<PathBuf>,

    /// Path to an eval dataset JSON file. If not provided, uses built-in scenarios.
    #[arg(long)]
    eval_dataset: Option<PathBuf>,

    /// Do not actually call any LLMs or agents — just parse and list fields.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq)]
enum AgentKind {
    /// Canned echo response (no real agent call). Useful for dry runs.
    Stub,
    /// POST to Aura's native web server endpoint.
    AuraHttp,
    /// POST to any OpenAI-compatible `/chat/completions` endpoint.
    OpenaiCompat,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq)]
enum MetricKind {
    /// Fuzzy substring match against expected_output (fast, offline).
    Fuzzy,
    /// LLM-as-judge scoring against task_description (slower, more meaningful).
    LlmJudge,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("aura_prompt_opt=info")
        .init();

    let cli = Cli::parse();

    // Resolve API key
    let api_key = cli.api_key.clone()
        .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .unwrap_or_default();

    if api_key.is_empty() && !cli.dry_run {
        eprintln!("Warning: No API key provided. Set --api-key, OPTIMIZER_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY.");
    }

    // Build LLM client (used by the optimizer's instructor and LlmJudge)
    let llm_client = LlmClient::new(&cli.base_url, &api_key, &cli.model);
    eprintln!("Optimizer LLM: {} via {}", cli.model, cli.base_url);

    // Load the TOML config (from file or extracted from values YAML)
    let (toml_str, original_yaml) = load_toml(&cli)?;

    // Parse config
    let mut config = aura_prompt_opt::parse_toml(&toml_str)?;
    eprintln!("Parsed config for agent: {}", config.agent.name);

    // Dynamic prompt discovery from Aura repo
    if let Some(ref aura_repo) = cli.aura_repo {
        match OrchestrationPrompts::from_aura_repo(aura_repo) {
            Ok(prompts) => {
                if let Some(ref mut orch) = config.orchestration {
                    eprintln!("Discovered {} prompts from {}", prompts.len(), aura_repo.display());
                    orch.prompts = prompts;
                }
            }
            Err(e) => eprintln!("Warning: could not discover prompts from {}: {e}", aura_repo.display()),
        }
    }

    // Show optimizable fields
    let fields = aura_prompt_opt::OptimizableField::all_fields(&config);
    eprintln!("\nOptimizable fields ({}):", fields.len());
    for field in &fields {
        eprintln!("  - {}", field.field_path());
    }

    // Build eval dataset
    let dataset = if let Some(ref path) = cli.eval_dataset {
        load_eval_dataset(path)?
    } else {
        default_eval_dataset(&config)
    };
    eprintln!("\nEval scenarios: {}", dataset.len());

    if cli.dry_run {
        eprintln!("\n[dry-run] Skipping optimization. Would run with:");
        eprintln!("  agent:  {:?}", cli.agent);
        eprintln!("  metric: {:?}", cli.metric);
        eprintln!("  fields: {}", fields.len());
        eprintln!("  scenarios: {}", dataset.len());
        return Ok(());
    }

    // Build agent runner
    let agent_runner = build_agent_runner(&cli)?;
    eprintln!("\nAgent runner: {}", agent_runner.name());

    // Build metrics
    let metrics: Vec<Box<dyn Metric>> = match cli.metric {
        MetricKind::Fuzzy => vec![Box::new(FuzzyMatch::new())],
        MetricKind::LlmJudge => vec![Box::new(
            LlmJudge::new(llm_client.clone()).with_pass_threshold(cli.pass_threshold),
        )],
    };
    eprintln!("Metric: {:?}", cli.metric);

    let runner = aura_prompt_opt::EvalRunner::new(metrics)
        .with_pass_threshold(cli.pass_threshold)
        .with_agent(agent_runner);

    let optimizer = AuraOptimizer::new(llm_client, runner)
        .with_pass_threshold(cli.pass_threshold);

    eprintln!("\nOptimizing...\n");
    let output = optimizer.optimize(&config, &dataset).await?;

    // Write outputs
    std::fs::write(&cli.output, &output.optimized_toml)?;
    eprintln!("Wrote optimized TOML:  {}", cli.output.display());

    std::fs::write(&cli.verbose_output, &output.verbose_toml)?;
    eprintln!("Wrote verbose TOML:    {}", cli.verbose_output.display());

    // If we loaded from a values file, inject the optimized TOML back
    if let (Some(ref yaml_str), Some(values_file_path)) = (&original_yaml, &cli.values_file) {
        let updated_yaml = aura_prompt_opt::values_file::inject_toml_into_yaml(
            yaml_str,
            &cli.config_key,
            &output.optimized_toml,
        )?;
        let out_path = match &cli.values_output {
            Some(p) => p.clone(),
            None => {
                let stem = values_file_path.file_stem().unwrap_or_default().to_string_lossy();
                values_file_path.with_file_name(format!("{stem}.optimized.yaml"))
            }
        };
        std::fs::write(&out_path, &updated_yaml)?;
        eprintln!("Wrote optimized YAML:  {}", out_path.display());
    }

    // Print verbose report
    println!("{}", output.verbose_toml);

    Ok(())
}

fn build_agent_runner(cli: &Cli) -> anyhow::Result<Box<dyn AgentRunner>> {
    match cli.agent {
        AgentKind::Stub => Ok(Box::new(StubAgentRunner::new())),
        AgentKind::AuraHttp => {
            let base = cli.agent_base_url.clone()
                .ok_or_else(|| anyhow::anyhow!("--agent-base-url required for aura-http"))?;
            let mut runner = AuraHttpRunner::new(base);
            if let Some(ref path) = cli.aura_chat_path {
                runner = runner.with_chat_path(path);
            }
            Ok(Box::new(runner))
        }
        AgentKind::OpenaiCompat => {
            let base = cli.agent_base_url.clone()
                .ok_or_else(|| anyhow::anyhow!("--agent-base-url required for openai-compat"))?;
            let key = cli.agent_api_key.clone().unwrap_or_default();
            Ok(Box::new(OpenAiCompatRunner::new(base, key)))
        }
    }
}

fn load_toml(cli: &Cli) -> anyhow::Result<(String, Option<String>)> {
    if let Some(ref values_path) = cli.values_file {
        let yaml_str = std::fs::read_to_string(values_path)?;
        let toml_str = aura_prompt_opt::values_file::extract_toml_from_yaml(
            &yaml_str,
            &cli.config_key,
        )?;
        eprintln!("Extracted TOML from {} (key: {})", values_path.display(), cli.config_key);
        Ok((toml_str, Some(yaml_str)))
    } else if let Some(ref config_path) = cli.config_file {
        let toml_str = std::fs::read_to_string(config_path)?;
        eprintln!("Loaded TOML from {}", config_path.display());
        Ok((toml_str, None))
    } else {
        anyhow::bail!("Provide either a TOML config file or --values-file <path>");
    }
}

fn load_eval_dataset(path: &Path) -> anyhow::Result<EvalDataset> {
    let content = std::fs::read_to_string(path)?;
    let dataset: EvalDataset = serde_json::from_str(&content)?;
    Ok(dataset)
}

fn default_eval_dataset(config: &aura_prompt_opt::AuraConfig) -> EvalDataset {
    let has_orchestration = config.orchestration.as_ref().map_or(false, |o| o.enabled);

    let mut dataset = EvalDataset::new("default-eval");

    if has_orchestration {
        dataset = dataset
            .add(EvalScenario::new(
                "parallel-diagnosis",
                "A deployment called payments-api is reporting elevated 5xx error rates. \
                 Check recent events in the payments namespace and pull the error_rate \
                 metric for the last 30 minutes. Tell me if this is an app problem or an infra problem.",
            ).with_task_description(
                "The response should: (1) reference findings from both a Kubernetes tool call \
                 and a Prometheus metric query, (2) quote specific values or events when citing \
                 evidence, (3) make a clear verdict (app vs infra vs unclear) backed by the evidence. \
                 Score 0 for no verdict or made-up data, 0.5 for partial coverage, 1.0 for a \
                 grounded multi-signal diagnosis."
            ))
            .add(EvalScenario::new(
                "single-worker-routing",
                "List all pods in the kube-system namespace.",
            ).with_task_description(
                "The response should route this to the cluster_inspector worker only, without \
                 invoking metrics_analyst or planning multi-step orchestration for a trivial \
                 single-tool task. Score 1.0 if the response contains a list of pods from \
                 kube-system, 0.5 if it routes correctly but adds unnecessary preamble, \
                 0 if it invokes Prometheus or refuses."
            ))
            .add(EvalScenario::new(
                "graceful-degradation",
                "What's the CPU usage for a service called 'totally-nonexistent-service-xyz'?",
            ).with_task_description(
                "The response should acknowledge that no metric matched, explain what was \
                 searched for, and not fabricate numbers. Score 1.0 for explicit 'not found' \
                 with query details, 0.5 for acknowledging no data but thin explanation, \
                 0 for hallucinating metrics."
            ))
            .add(EvalScenario::new(
                "cross-domain-synthesis",
                "Is the payments-api deployment healthy right now? Use whatever signals \
                 are available.",
            ).with_task_description(
                "The response should synthesize evidence from multiple sources (at least \
                 pod status from cluster_inspector AND metrics from metrics_analyst) into \
                 a single cohesive answer, not two disjoint sections. Score 1.0 for \
                 integrated synthesis, 0.5 for both sources but stitched-together prose, \
                 0 for single-source or no data."
            ));
    } else {
        dataset = dataset.add(EvalScenario::new(
            "basic-diagnosis",
            "Help me diagnose a performance issue with our web application. \
             Users report slow page loads during peak hours.",
        ).with_task_description(
            "The response should outline specific diagnostic steps in a prioritized order, \
             mentioning concrete signals to check (CPU, memory, DB, network). Score 1.0 for \
             a structured prioritized list with reasoning, 0.5 for a vague list, 0 for \
             generic non-actionable advice."
        ));
    }

    dataset
}
