use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatResponseMessage {
    content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Clone, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

/// Default base URL for the LLM API (OpenAI).
///
/// To use OpenRouter or another OpenAI-compatible provider, set the
/// `OPTIMIZER_BASE_URL` environment variable (e.g. `https://openrouter.ai/api/v1`).
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Default model name used when none is specified.
pub const DEFAULT_MODEL: &str = "gpt-4o";

/// OpenAI-compatible chat completions client.
///
/// Configure via environment variables:
///   OPTIMIZER_API_KEY   (or OPENAI_API_KEY, OPENROUTER_API_KEY)
///   OPTIMIZER_BASE_URL  (default: https://api.openai.com/v1)
///   OPTIMIZER_MODEL     (default: gpt-4o)
///
/// To use OpenRouter:
///   OPTIMIZER_BASE_URL=https://openrouter.ai/api/v1
///   OPTIMIZER_API_KEY=sk-or-v1-...
///   OPTIMIZER_MODEL=openai/gpt-4o  (or any OpenRouter model slug)
#[derive(Debug, Clone)]
pub struct LlmClient {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    client: reqwest::Client,
}

impl LlmClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Self {
        let base_url = std::env::var("OPTIMIZER_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        let api_key = std::env::var("OPTIMIZER_API_KEY")
            .or_else(|_| std::env::var("OPENROUTER_API_KEY"))
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .unwrap_or_default();
        let model = std::env::var("OPTIMIZER_MODEL")
            .unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        Self::new(base_url, api_key, model)
    }

    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        self.chat_with_options(messages, None, None).await
    }

    pub async fn chat_with_options(
        &self,
        messages: &[ChatMessage],
        temperature: Option<f64>,
        max_tokens: Option<u32>,
    ) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let request_body = ChatCompletionRequest {
            model: &self.model,
            messages,
            temperature,
            max_tokens,
        };

        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&request_body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(Error::LlmResponse(format!("HTTP {status}: {body}")));
        }

        let completion: ChatCompletionResponse = resp.json().await?;
        let content = completion
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| Error::LlmResponse("empty response from LLM".into()))?;

        Ok(content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_constructors() {
        let sys = ChatMessage::system("you are a bot");
        assert_eq!(sys.role, "system");
        assert_eq!(sys.content, "you are a bot");

        let user = ChatMessage::user("hello");
        assert_eq!(user.role, "user");

        let asst = ChatMessage::assistant("hi there");
        assert_eq!(asst.role, "assistant");
    }

    #[test]
    fn test_client_from_env_uses_defaults() {
        std::env::remove_var("OPTIMIZER_BASE_URL");
        std::env::remove_var("OPTIMIZER_MODEL");
        std::env::remove_var("OPTIMIZER_API_KEY");
        std::env::remove_var("OPENROUTER_API_KEY");
        std::env::remove_var("OPENAI_API_KEY");

        let client = LlmClient::from_env();
        assert_eq!(client.base_url, DEFAULT_BASE_URL);
        assert_eq!(client.base_url, "https://api.openai.com/v1");
        assert_eq!(client.model, "gpt-4o");
    }

    #[test]
    fn test_client_from_env_reads_env_vars() {
        std::env::set_var("OPTIMIZER_BASE_URL", "http://localhost:8080/v1");
        std::env::set_var("OPTIMIZER_MODEL", "claude-sonnet-4-6");
        std::env::set_var("OPTIMIZER_API_KEY", "test-key");

        let client = LlmClient::from_env();
        assert_eq!(client.base_url, "http://localhost:8080/v1");
        assert_eq!(client.model, "claude-sonnet-4-6");
        assert_eq!(client.api_key, "test-key");

        std::env::remove_var("OPTIMIZER_BASE_URL");
        std::env::remove_var("OPTIMIZER_MODEL");
        std::env::remove_var("OPTIMIZER_API_KEY");
    }
}
