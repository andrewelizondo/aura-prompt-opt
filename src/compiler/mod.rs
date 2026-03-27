pub mod optimized;
pub mod verbose;

pub use optimized::compile_optimized;
pub use verbose::compile_verbose;

/// Output of the compilation stage.
#[derive(Debug, Clone)]
pub struct CompilationOutput {
    pub optimized_toml: String,
    pub verbose_toml: String,
}
