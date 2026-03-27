use crate::error::Result;
use crate::eval::scenario::EvalScenario;
use crate::llm::LlmClient;
use crate::llm::client::ChatMessage;

#[derive(Debug, Clone)]
pub struct MetricScore {
    pub score: f64,
    pub rationale: String,
    pub passed: bool,
}

pub trait Metric: Send + Sync {
    fn name(&self) -> &str;
    fn score<'a>(
        &'a self,
        scenario: &'a EvalScenario,
        output: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<MetricScore>> + Send + 'a>>;
}

pub struct ExactMatch {
    pub pass_threshold: f64,
}

impl ExactMatch {
    pub fn new() -> Self {
        Self { pass_threshold: 1.0 }
    }
}

impl Default for ExactMatch {
    fn default() -> Self { Self::new() }
}

impl Metric for ExactMatch {
    fn name(&self) -> &str { "exact_match" }

    fn score<'a>(
        &'a self,
        scenario: &'a EvalScenario,
        output: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<MetricScore>> + Send + 'a>> {
        Box::pin(async move {
            let expected = scenario.expected_output.as_deref().unwrap_or("");
            let matches = output.trim() == expected.trim();
            Ok(MetricScore {
                score: if matches { 1.0 } else { 0.0 },
                rationale: if matches {
                    "Exact match".to_string()
                } else {
                    format!("Expected: {:?}, got: {:?}", expected, output.trim())
                },
                passed: matches,
            })
        })
    }
}

pub struct FuzzyMatch {
    pub pass_threshold: f64,
    pub case_sensitive: bool,
}

impl FuzzyMatch {
    pub fn new() -> Self {
        Self { pass_threshold: 1.0, case_sensitive: false }
    }
}

impl Default for FuzzyMatch {
    fn default() -> Self { Self::new() }
}

impl Metric for FuzzyMatch {
    fn name(&self) -> &str { "fuzzy_match" }

    fn score<'a>(
        &'a self,
        scenario: &'a EvalScenario,
        output: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<MetricScore>> + Send + 'a>> {
        let case_sensitive = self.case_sensitive;
        Box::pin(async move {
            let expected = scenario.expected_output.as_deref().unwrap_or("");
            let found = if case_sensitive {
                output.contains(expected)
            } else {
                output.to_lowercase().contains(&expected.to_lowercase())
            };
            Ok(MetricScore {
                score: if found { 1.0 } else { 0.0 },
                rationale: if found {
                    format!("Output contains expected substring: {:?}", expected)
                } else {
                    format!("Expected substring {:?} not found in output", expected)
                },
                passed: found,
            })
        })
    }
}

pub struct LlmJudge {
    pub client: LlmClient,
    pub pass_threshold: f64,
}

impl LlmJudge {
    pub fn new(client: LlmClient) -> Self {
        Self { client, pass_threshold: 0.7 }
    }

    pub fn with_pass_threshold(mut self, threshold: f64) -> Self {
        self.pass_threshold = threshold;
        self
    }
}

impl Metric for LlmJudge {
    fn name(&self) -> &str { "llm_judge" }

    fn score<'a>(
        &'a self,
        scenario: &'a EvalScenario,
        output: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<MetricScore>> + Send + 'a>> {
        Box::pin(async move {
            let task = scenario.task_description.as_deref()
                .unwrap_or("complete the given task correctly");

            let prompt = format!(
                "You are evaluating an AI agent's response.\n\n\
                 Task description: {task}\n\n\
                 User input: {input}\n\n\
                 Agent output:\n{output}\n\n\
                 Score the output from 0 to 10 on how well it completes the task.\n\
                 Respond with ONLY a JSON object: {{\"score\": <0-10>, \"rationale\": \"<brief explanation>\"}}",
                task = task,
                input = scenario.input,
                output = output,
            );

            let messages = vec![
                ChatMessage::system("You are a precise evaluator. Return only valid JSON."),
                ChatMessage::user(prompt),
            ];

            let response = self.client.chat(&messages).await?;
            let (score, rationale) = parse_judge_response(&response)?;
            let normalized = score / 10.0;

            Ok(MetricScore {
                score: normalized,
                rationale,
                passed: normalized >= self.pass_threshold,
            })
        })
    }
}

fn parse_judge_response(response: &str) -> Result<(f64, String)> {
    let json_str = extract_json(response);
    let v: serde_json::Value = serde_json::from_str(&json_str)
        .map_err(|e| crate::error::Error::LlmResponse(format!("judge JSON parse error: {e}")))?;

    let score = v["score"]
        .as_f64()
        .ok_or_else(|| crate::error::Error::LlmResponse("missing 'score' field".into()))?;

    let rationale = v["rationale"]
        .as_str()
        .unwrap_or("no rationale provided")
        .to_string();

    if !(0.0..=10.0).contains(&score) {
        return Err(crate::error::Error::LlmResponse(
            format!("score {score} out of range [0, 10]"),
        ));
    }

    Ok((score, rationale))
}

fn extract_json(s: &str) -> String {
    let s = s.trim();
    let s = s.strip_prefix("```json").unwrap_or(s);
    let s = s.strip_prefix("```").unwrap_or(s);
    let s = s.strip_suffix("```").unwrap_or(s);
    s.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::scenario::EvalScenario;

    #[tokio::test]
    async fn test_exact_match_pass() {
        let metric = ExactMatch::new();
        let scenario = EvalScenario::new("test", "q").with_expected("42");
        let result = metric.score(&scenario, "42").await.unwrap();
        assert_eq!(result.score, 1.0);
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_exact_match_fail() {
        let metric = ExactMatch::new();
        let scenario = EvalScenario::new("test", "q").with_expected("42");
        let result = metric.score(&scenario, "43").await.unwrap();
        assert_eq!(result.score, 0.0);
        assert!(!result.passed);
    }

    #[tokio::test]
    async fn test_fuzzy_match_case_insensitive() {
        let metric = FuzzyMatch::new();
        let scenario = EvalScenario::new("test", "q").with_expected("hello");
        let result = metric.score(&scenario, "Say HELLO world").await.unwrap();
        assert_eq!(result.score, 1.0);
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_fuzzy_match_miss() {
        let metric = FuzzyMatch::new();
        let scenario = EvalScenario::new("test", "q").with_expected("goodbye");
        let result = metric.score(&scenario, "Hello world").await.unwrap();
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn test_parse_judge_response_valid() {
        let resp = r#"{"score": 8, "rationale": "good answer"}"#;
        let (score, rationale) = parse_judge_response(resp).unwrap();
        assert_eq!(score, 8.0);
        assert_eq!(rationale, "good answer");
    }

    #[test]
    fn test_parse_judge_response_with_markdown() {
        let resp = "```json\n{\"score\": 7, \"rationale\": \"ok\"}\n```";
        let (score, _) = parse_judge_response(resp).unwrap();
        assert_eq!(score, 7.0);
    }

    #[test]
    fn test_parse_judge_response_out_of_range() {
        let resp = r#"{"score": 11, "rationale": "perfect"}"#;
        assert!(parse_judge_response(resp).is_err());
    }
}
