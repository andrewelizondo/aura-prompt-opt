use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("TOML parse error: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("TOML serialize error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    #[error("LLM request failed: {0}")]
    LlmRequest(#[from] reqwest::Error),

    #[error("LLM response malformed: {0}")]
    LlmResponse(String),

    #[error("Evaluation error: {0}")]
    Eval(String),

    #[error("Optimization error: {0}")]
    Optimization(String),

    #[error("Compilation error: {0}")]
    Compilation(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, Error>;
