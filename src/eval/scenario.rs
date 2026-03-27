use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalScenario {
    pub name: String,
    pub input: String,
    pub expected_output: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub task_description: Option<String>,
}

impl EvalScenario {
    pub fn new(name: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            input: input.into(),
            expected_output: None,
            tags: Vec::new(),
            task_description: None,
        }
    }

    pub fn with_expected(mut self, expected: impl Into<String>) -> Self {
        self.expected_output = Some(expected.into());
        self
    }

    pub fn with_task_description(mut self, desc: impl Into<String>) -> Self {
        self.task_description = Some(desc.into());
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalDataset {
    pub name: String,
    pub scenarios: Vec<EvalScenario>,
}

impl EvalDataset {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), scenarios: Vec::new() }
    }

    pub fn add(mut self, scenario: EvalScenario) -> Self {
        self.scenarios.push(scenario);
        self
    }

    pub fn len(&self) -> usize {
        self.scenarios.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scenarios.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_builder() {
        let scenario = EvalScenario::new("test", "what is 2+2?")
            .with_expected("4")
            .with_task_description("basic math")
            .with_tags(vec!["math".into()]);

        assert_eq!(scenario.name, "test");
        assert_eq!(scenario.input, "what is 2+2?");
        assert_eq!(scenario.expected_output.as_deref(), Some("4"));
        assert_eq!(scenario.tags, vec!["math"]);
    }

    #[test]
    fn test_dataset_builder() {
        let dataset = EvalDataset::new("my-dataset")
            .add(EvalScenario::new("s1", "hello"))
            .add(EvalScenario::new("s2", "world"));

        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());
    }
}
