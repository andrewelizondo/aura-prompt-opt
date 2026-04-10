use aura_prompt_opt::{
    AuraOptimizer, EvalDataset, EvalScenario, FuzzyMatch, LlmClient, Metric,
    config::schema::OrchestrationPrompts,
};
use clap::Parser;
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

    /// LLM API base URL (OpenAI-compatible).
    #[arg(long, env = "OPTIMIZER_BASE_URL", default_value = "https://openrouter.ai/api/v1")]
    base_url: String,

    /// LLM API key.
    #[arg(long, env = "OPTIMIZER_API_KEY")]
    api_key: Option<String>,

    /// LLM model name.
    #[arg(long, env = "OPTIMIZER_MODEL", default_value = "openai/gpt-4o")]
    model: String,

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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("aura_prompt_opt=info")
        .init();

    let cli = Cli::parse();

    // Resolve API key
    let api_key = cli.api_key.clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .unwrap_or_default();

    if api_key.is_empty() {
        eprintln!("Warning: No API key provided. Set --api-key, OPTIMIZER_API_KEY, or OPENAI_API_KEY.");
    }

    // Build LLM client
    let llm_client = LlmClient::new(&cli.base_url, &api_key, &cli.model);
    eprintln!("LLM: {} via {}", cli.model, cli.base_url);

    // Load the TOML config (from file or extracted from values YAML)
    let (toml_str, original_yaml) = load_toml(&cli)?;

    // Parse config
    let mut config = aura_prompt_opt::parse_toml(&toml_str)?;
    eprintln!("Parsed config for agent: {}", config.agent.name);

    // Dynamic prompt discovery from Aura repo
    if let Some(ref aura_repo) = cli.aura_repo {
        if let Ok(prompts) = OrchestrationPrompts::from_aura_repo(aura_repo) {
            if let Some(ref mut orch) = config.orchestration {
                eprintln!("Discovered {} prompts from {}", prompts.len(), aura_repo.display());
                orch.prompts = prompts;
            }
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

    // Run optimization
    let metrics: Vec<Box<dyn Metric>> = vec![Box::new(FuzzyMatch::new())];
    let runner = aura_prompt_opt::EvalRunner::new(metrics);
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
                "multi-step-query",
                "What's the average response time for the /api/users endpoint \
                 over the last hour, and are there any related slow queries?",
            ).with_task_description(
                "Agent should decompose into parallel tasks and synthesize results",
            ))
            .add(EvalScenario::new(
                "single-worker-routing",
                "List all pods in the kube-system namespace.",
            ).with_task_description(
                "Agent should route to a single worker without unnecessary orchestration",
            ))
            .add(EvalScenario::new(
                "error-handling",
                "Get the CPU usage for a service called 'nonexistent-service'.",
            ).with_task_description(
                "Agent should handle missing data gracefully",
            ));
    } else {
        dataset = dataset
            .add(EvalScenario::new(
                "basic-query",
                "Help me diagnose a performance issue with our web application.",
            ).with_task_description(
                "Agent should provide structured diagnostic steps",
            ));
    }

    dataset
}
