//! Example: optimize an orchestration-mode agent config.
//!
//! This demonstrates how aura-prompt-opt optimizes both the agent system prompt
//! AND the orchestration prompt templates (coordinator, worker, synthesis,
//! evaluation, reflection, phase continuation, etc.).
//!
//! Prompts can be dynamically discovered from a local Aura checkout, so when
//! Aura adds/changes prompts, the optimizer picks them up automatically.
//!
//! Usage:
//!   OPTIMIZER_API_KEY=sk-... cargo run --example optimize_orchestration
//!
//! With dynamic discovery from a local Aura repo:
//!   AURA_REPO=/path/to/aura OPTIMIZER_API_KEY=sk-... cargo run --example optimize_orchestration

use aura_prompt_opt::{
    AuraOptimizer, EvalDataset, EvalScenario, FuzzyMatch, OptimizableField,
    config::schema::OrchestrationPrompts,
};
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("aura_prompt_opt=debug,info")
        .init();

    // ── Dynamic prompt discovery ───────────────────────────────────────
    //
    // If AURA_REPO is set, discover prompts from the local checkout.
    // Otherwise, use the embedded defaults compiled into the optimizer.
    if let Ok(aura_repo) = std::env::var("AURA_REPO") {
        let repo_path = Path::new(&aura_repo);
        match aura_prompt_opt::discover_from_aura_repo(repo_path) {
            Ok(discovered) => {
                println!("Discovered {} prompts from {}", discovered.len(), aura_repo);
                for p in &discovered {
                    println!(
                        "  - {} ({} template vars, {} bytes)",
                        p.name,
                        p.template_vars.len(),
                        p.content.len(),
                    );
                }
            }
            Err(e) => {
                eprintln!("Warning: could not discover prompts from {aura_repo}: {e}");
                eprintln!("Falling back to embedded defaults.");
            }
        }
    }

    // Build evaluation scenarios for orchestration-mode behaviors
    let dataset = EvalDataset::new("orchestration-eval")
        .add(
            EvalScenario::new(
                "multi-step-query",
                "What's the average response time for the /api/users endpoint \
                 over the last hour, and are there any related database slow queries?",
            )
            .with_task_description(
                "Agent should decompose this into parallel tasks: one for metrics \
                 querying and one for database slow-query analysis, then synthesize results",
            )
            .with_tags(vec!["orchestration".into(), "planning".into()]),
        )
        .add(
            EvalScenario::new(
                "worker-delegation",
                "Show me the schema for the users table and any recent alerts on the users service.",
            )
            .with_task_description(
                "Agent should route database schema inspection to the database worker \
                 and alert queries to the monitoring worker",
            )
            .with_tags(vec!["orchestration".into(), "routing".into()]),
        )
        .add(
            EvalScenario::new(
                "synthesis-quality",
                "Compare the p99 latency of /api/users and /api/orders over the last 24 hours.",
            )
            .with_task_description(
                "Agent should query metrics for both endpoints, then synthesize \
                 a coherent comparison rather than just concatenating results",
            )
            .with_tags(vec!["orchestration".into(), "synthesis".into()]),
        )
        .add(
            EvalScenario::new(
                "single-worker-task",
                "Run SELECT count(*) FROM orders WHERE status = 'pending';",
            )
            .with_task_description(
                "Agent should route this directly to the database worker without \
                 unnecessary orchestration overhead",
            )
            .with_tags(vec!["routing".into(), "simple".into()]),
        )
        .add(
            EvalScenario::new(
                "error-recovery",
                "Get the CPU usage for a service called 'nonexistent-service'.",
            )
            .with_task_description(
                "Agent should handle the case where the monitoring query returns \
                 no data gracefully, reporting the gap clearly",
            )
            .with_tags(vec!["orchestration".into(), "error-handling".into()]),
        );

    println!("Loaded {} evaluation scenarios", dataset.len());

    // Load config, optionally with prompts discovered from Aura repo
    let config_path = Path::new("examples/sample_configs/orchestration_agent.toml");
    let toml_str = std::fs::read_to_string(config_path)?;
    let mut config = aura_prompt_opt::parse_toml(&toml_str)?;

    // If AURA_REPO is set, overlay discovered prompts onto the config
    if let Ok(aura_repo) = std::env::var("AURA_REPO") {
        if let Ok(prompts) = OrchestrationPrompts::from_aura_repo(Path::new(&aura_repo)) {
            if let Some(ref mut orch) = config.orchestration {
                println!(
                    "\nLoaded {} prompts from Aura repo (vs {} embedded defaults)",
                    prompts.len(),
                    orch.prompts.len(),
                );
                orch.prompts = prompts;
            }
        }
    }

    // Show which fields will be optimized
    let fields = OptimizableField::all_fields(&config);
    println!("\nOptimizable fields ({} total):", fields.len());
    for field in &fields {
        println!("  - {} ({})", field.field_path(), field.description_for(&config));
    }

    // Run optimization
    let metrics: Vec<Box<dyn aura_prompt_opt::Metric>> = vec![Box::new(FuzzyMatch::new())];
    let optimizer = AuraOptimizer::from_env(metrics).with_pass_threshold(0.6);

    println!("\nStarting optimization of {}...", config_path.display());

    match optimizer.optimize(&config, &dataset).await {
        Ok(output) => {
            println!("\n=== OPTIMIZED TOML ===\n");
            println!("{}", output.optimized_toml);

            println!("\n=== VERBOSE TOML ===\n");
            println!("{}", output.verbose_toml);

            std::fs::write("config.optimized.toml", &output.optimized_toml)?;
            std::fs::write("config.verbose.toml", &output.verbose_toml)?;
            println!("\nSaved: config.optimized.toml and config.verbose.toml");
        }
        Err(e) => {
            eprintln!("Optimization failed: {e}");
            std::process::exit(1);
        }
    }

    Ok(())
}
