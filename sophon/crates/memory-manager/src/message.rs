use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sophon_core::tokens::count_tokens;
use uuid::Uuid;

use crate::index::SemanticIndex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub token_count: usize,
    pub id: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        let content = content.into();
        Self {
            role,
            token_count: count_tokens(&content),
            content,
            timestamp: Utc::now(),
            id: Uuid::new_v4().to_string(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedMemory {
    pub summary: String,
    pub stable_facts: Vec<Fact>,
    pub recent_messages: Vec<Message>,
    pub index: SemanticIndex,
    pub token_count: usize,
    pub original_message_count: usize,
}

impl CompressedMemory {
    pub fn to_system_context(&self) -> String {
        let mut output = String::new();

        output.push_str("<conversation_summary>\n");
        output.push_str(&self.summary);
        output.push_str("\n</conversation_summary>\n\n");

        if !self.stable_facts.is_empty() {
            output.push_str("<established_facts>\n");
            for fact in &self.stable_facts {
                if !fact.superseded {
                    output.push_str(&format!("- {}\n", fact.content));
                }
            }
            output.push_str("</established_facts>\n\n");
        }

        output
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub content: String,
    pub established_at: usize,
    pub superseded: bool,
    pub category: FactCategory,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FactCategory {
    UserIdentity,
    ProjectContext,
    TechnicalDecision,
    Instruction,
    Constraint,
}
