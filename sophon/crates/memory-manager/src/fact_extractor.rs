use once_cell::sync::Lazy;
use regex::Regex;

use crate::message::{Fact, FactCategory, Message, Role};

struct FactPattern {
    regex: Regex,
    category: FactCategory,
    template: &'static str,
}

static FACT_PATTERNS: Lazy<Vec<FactPattern>> = Lazy::new(|| {
    vec![
        FactPattern {
            regex: Regex::new(r"(?i)my name is ([A-Za-z][\w-]*)").expect("valid regex"),
            category: FactCategory::UserIdentity,
            template: "User's name is {1}",
        },
        FactPattern {
            regex: Regex::new(r"(?i)i(?:'m| am) (?:a |an )?([A-Za-z][\w\s]{1,40})")
                .expect("valid regex"),
            category: FactCategory::UserIdentity,
            template: "User is {1}",
        },
        FactPattern {
            regex: Regex::new(r"(?i)(?:working on|building|creating) (?:a |an )?(.+?)(?:\.|$)")
                .expect("valid regex"),
            category: FactCategory::ProjectContext,
            template: "Working on: {1}",
        },
        FactPattern {
            regex: Regex::new(r"(?i)(?:using|chose|decided on|going with) ([A-Za-z][\w\s]{1,40})")
                .expect("valid regex"),
            category: FactCategory::TechnicalDecision,
            template: "Using {1}",
        },
        FactPattern {
            regex: Regex::new(r"(?i)(?:always|never|please|don't) (.+?)(?:\.|$)").expect("valid regex"),
            category: FactCategory::Instruction,
            template: "Instruction: {0}",
        },
    ]
});

/// Extract stable facts from user-authored messages.
pub fn extract_facts(messages: &[Message]) -> Vec<Fact> {
    let mut facts = Vec::new();

    for (idx, message) in messages.iter().enumerate() {
        if message.role != Role::User {
            continue;
        }

        for pattern in FACT_PATTERNS.iter() {
            for cap in pattern.regex.captures_iter(&message.content) {
                let full = cap.get(0).map(|m| m.as_str().trim()).unwrap_or_default();
                let group1 = cap.get(1).map(|m| m.as_str().trim()).unwrap_or_default();
                let content = pattern
                    .template
                    .replace("{0}", full)
                    .replace("{1}", group1)
                    .trim()
                    .to_string();

                if content.is_empty() {
                    continue;
                }

                if !facts.iter().any(|f: &Fact| f.content.eq_ignore_ascii_case(&content)) {
                    facts.push(Fact {
                        content,
                        established_at: idx,
                        superseded: false,
                        category: pattern.category,
                    });
                }
            }
        }
    }

    mark_superseded(&mut facts);
    facts
}

fn mark_superseded(facts: &mut [Fact]) {
    for i in 0..facts.len() {
        for j in (i + 1)..facts.len() {
            if facts[i].category == facts[j].category && facts[i].content != facts[j].content {
                // Keep latest statement active for mutable categories.
                if matches!(
                    facts[i].category,
                    FactCategory::UserIdentity
                        | FactCategory::ProjectContext
                        | FactCategory::TechnicalDecision
                ) {
                    if facts[i].established_at < facts[j].established_at {
                        facts[i].superseded = true;
                    } else {
                        facts[j].superseded = true;
                    }
                }
            }
        }
    }
}
