/// Agent execution abstraction.
///
/// The optimizer needs to actually run the agent being optimized to score
/// its outputs. `AgentRunner` is the injection point that decouples the
/// eval pipeline from any specific agent implementation.
///
/// Implementations:
/// - `StubAgentRunner` — canned echo response (for tests / dry-runs)
/// - `HttpAgentRunner` — POST to a running Aura web server
/// - `OpenAiCompatRunner` — POST to any OpenAI-compatible chat endpoint
///   (useful when Aura is deployed behind an OpenAI-compatible gateway)

use crate::config::schema::AuraConfig;
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

/// Trait for executing an agent against an input query.
///
/// Boxed for dynamic dispatch so `EvalRunner` can swap implementations
/// at runtime (configured via CLI flags or library users).
pub trait AgentRunner: Send + Sync {
    fn name(&self) -> &str;

    /// Runs the agent with the given config and input, returning the
    /// agent's response text.
    fn run<'a>(
        &'a self,
        config: &'a AuraConfig,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;
}

// ── Stub ───────────────────────────────────────────────────────────────

/// A stub runner that echoes a synthetic response. Useful for tests
/// and for smoke-testing the pipeline without a real agent.
pub struct StubAgentRunner;

impl StubAgentRunner {
    pub fn new() -> Self { Self }
}

impl Default for StubAgentRunner {
    fn default() -> Self { Self::new() }
}

impl AgentRunner for StubAgentRunner {
    fn name(&self) -> &str { "stub" }

    fn run<'a>(
        &'a self,
        config: &'a AuraConfig,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        let name = config.agent.name.clone();
        let input = input.to_string();
        Box::pin(async move {
            Ok(format!("[STUB] Agent '{name}' received: {input}"))
        })
    }
}

// ── OpenAI-compatible HTTP runner ──────────────────────────────────────

/// Runs the agent via an OpenAI-compatible `/chat/completions` endpoint.
///
/// This fits any deployment where the Aura web server exposes an
/// OpenAI-compatible chat API (e.g. behind a gateway or via a compatible
/// adapter). Each eval scenario makes one POST request.
pub struct OpenAiCompatRunner {
    base_url: String,
    api_key: String,
    model_override: Option<String>,
    timeout: Duration,
    client: reqwest::Client,
}

impl OpenAiCompatRunner {
    /// Create a new runner. Pass `None` for `model_override` to use the
    /// model from the agent config being optimized.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model_override: None,
            timeout: Duration::from_secs(120),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model_override = Some(model.into());
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMsg<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Serialize)]
struct ChatMsg<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMsg,
}

#[derive(Deserialize)]
struct ChatResponseMsg {
    content: Option<String>,
}

impl AgentRunner for OpenAiCompatRunner {
    fn name(&self) -> &str { "openai-compat" }

    fn run<'a>(
        &'a self,
        config: &'a AuraConfig,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let model = self.model_override.as_deref()
                .unwrap_or_else(|| config.llm.model_name());

            let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
            let req = ChatRequest {
                model,
                messages: vec![
                    ChatMsg { role: "system", content: &config.agent.system_prompt },
                    ChatMsg { role: "user", content: input },
                ],
                temperature: config.agent.temperature,
                max_tokens: config.agent.max_tokens.map(|n| n as u32),
            };

            let resp = self.client
                .post(&url)
                .bearer_auth(&self.api_key)
                .json(&req)
                .send()
                .await
                .map_err(|e| Error::Eval(format!("agent HTTP request failed: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                let truncated: String = body.chars().take(500).collect();
                return Err(Error::Eval(format!("agent returned HTTP {status}: {truncated}")));
            }

            let completion: ChatResponse = resp.json().await
                .map_err(|e| Error::Eval(format!("failed to parse agent response: {e}")))?;

            completion.choices.into_iter().next()
                .and_then(|c| c.message.content)
                .ok_or_else(|| Error::Eval("agent returned empty response".into()))
        })
    }
}

// ── Aura HTTP runner ───────────────────────────────────────────────────

/// Runs the agent via the Aura web server's native HTTP API.
///
/// Aura's `aura-web-server` exposes a streaming chat endpoint. This runner
/// sends a single query and collects the final response text.
///
/// The exact request/response shape depends on the Aura version. This
/// implementation uses a best-effort POST to `{base_url}/v1/chat` with
/// `{"query": "..."}` and expects a JSON response with a `response` field.
/// Override via env var `AURA_CHAT_PATH` if your deployment uses a
/// different path.
pub struct AuraHttpRunner {
    base_url: String,
    chat_path: String,
    timeout: Duration,
    client: reqwest::Client,
}

impl AuraHttpRunner {
    pub fn new(base_url: impl Into<String>) -> Self {
        let chat_path = std::env::var("AURA_CHAT_PATH")
            .unwrap_or_else(|_| "/v1/chat".to_string());
        Self {
            base_url: base_url.into(),
            chat_path,
            timeout: Duration::from_secs(180),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_chat_path(mut self, path: impl Into<String>) -> Self {
        self.chat_path = path.into();
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[derive(Serialize)]
struct AuraChatRequest<'a> {
    query: &'a str,
}

impl AgentRunner for AuraHttpRunner {
    fn name(&self) -> &str { "aura-http" }

    fn run<'a>(
        &'a self,
        _config: &'a AuraConfig,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let url = format!(
                "{}{}",
                self.base_url.trim_end_matches('/'),
                self.chat_path,
            );
            let req = AuraChatRequest { query: input };

            let resp = self.client
                .post(&url)
                .json(&req)
                .send()
                .await
                .map_err(|e| Error::Eval(format!("aura HTTP request failed: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                let truncated: String = body.chars().take(500).collect();
                return Err(Error::Eval(format!("aura returned HTTP {status}: {truncated}")));
            }

            // Try to extract the response text from common JSON shapes
            let body = resp.text().await
                .map_err(|e| Error::Eval(format!("failed to read aura response: {e}")))?;

            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                // Try common field names
                for field in &["response", "result", "text", "output", "content", "message"] {
                    if let Some(s) = v.get(field).and_then(|x| x.as_str()) {
                        return Ok(s.to_string());
                    }
                }
                // Fall back to the whole JSON as text
                return Ok(body);
            }

            // Non-JSON response — return as-is
            Ok(body)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::{AgentConfig, AuraConfig};

    #[tokio::test]
    async fn test_stub_agent_runner_returns_canned_response() {
        let runner = StubAgentRunner::new();
        let config = AuraConfig {
            agent: AgentConfig { name: "TestAgent".into(), ..AgentConfig::default() },
            ..AuraConfig::default()
        };
        let output = runner.run(&config, "hello").await.unwrap();
        assert!(output.contains("TestAgent"));
        assert!(output.contains("hello"));
        assert_eq!(runner.name(), "stub");
    }

    #[test]
    fn test_runners_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StubAgentRunner>();
        assert_send_sync::<OpenAiCompatRunner>();
        assert_send_sync::<AuraHttpRunner>();
        assert_send_sync::<Box<dyn AgentRunner>>();
    }

    #[test]
    fn test_openai_runner_constructors() {
        let runner = OpenAiCompatRunner::new("http://localhost:8080/v1", "sk-test")
            .with_model("gpt-4o")
            .with_timeout(Duration::from_secs(30));
        assert_eq!(runner.name(), "openai-compat");
        assert_eq!(runner.base_url, "http://localhost:8080/v1");
    }

    #[test]
    fn test_aura_runner_custom_path() {
        let runner = AuraHttpRunner::new("http://aura:8080")
            .with_chat_path("/api/chat");
        assert_eq!(runner.chat_path, "/api/chat");
    }
}
