//! `k3-tokenizer`: Shared tokenizer, chat template, and streaming piece decoder.
//!
//! Provides decoupled tokenization interfaces shared between `k3-diag` and
//! `k3-manager` per ADR-025 without dragging in engine compute or hardware
//! affinity dependencies.

use std::collections::HashMap;

use k3_core::K3Error;

/// Special control token IDs extracted from model metadata.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SpecialTokens {
    /// Beginning-of-sequence token ID.
    pub bos_token_id: Option<u32>,
    /// End-of-sequence token ID.
    pub eos_token_id: Option<u32>,
    /// Padding token ID.
    pub padding_token_id: Option<u32>,
    /// Whether the prompt should automatically prepend `bos_token_id`.
    pub add_bos_token: bool,
    /// Explicit stop sequence token IDs.
    pub stop_tokens: Vec<u32>,
}

impl SpecialTokens {
    /// Checks if a given token ID is considered a sequence stop token.
    #[must_use]
    pub fn is_stop(&self, token_id: u32) -> bool {
        if self.eos_token_id == Some(token_id) {
            return true;
        }
        self.stop_tokens.contains(&token_id)
    }
}

/// Tokenizer abstraction for text encoding and real-time token streaming.
pub trait Tokenizer: Send + Sync + std::fmt::Debug {
    /// Encodes raw text into a sequence of token IDs.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if characters cannot be encoded.
    fn encode(&self, text: &str) -> Result<Vec<u32>, K3Error>;

    /// Decodes a sequence of token IDs back into text.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if token IDs cannot be resolved.
    fn decode(&self, tokens: &[u32]) -> Result<String, K3Error>;

    /// Decodes a single token ID into its text fragment for streaming output.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if the token ID is not recognized.
    fn decode_piece(&self, token_id: u32) -> Result<Option<String>, K3Error>;

    /// Returns the active special tokens configuration.
    fn special_tokens(&self) -> &SpecialTokens;
}

/// Baseline in-memory vocabulary tokenizer.
#[derive(Debug, Clone, Default)]
pub struct SimpleTokenizer {
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    special_tokens: SpecialTokens,
}

impl SimpleTokenizer {
    /// Creates an empty tokenizer with default special tokens.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the special tokens configuration.
    pub fn set_special_tokens(&mut self, special_tokens: SpecialTokens) {
        self.special_tokens = special_tokens;
    }

    /// Registers a token string and corresponding token ID.
    pub fn add_token(&mut self, token: impl Into<String>, id: u32) {
        let text = token.into();
        self.token_to_id.insert(text.clone(), id);
        self.id_to_token.insert(id, text);
    }

    /// Number of vocabulary items currently registered.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }
}

impl Tokenizer for SimpleTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, K3Error> {
        let mut tokens = Vec::new();
        if self.special_tokens.add_bos_token {
            if let Some(bos) = self.special_tokens.bos_token_id {
                tokens.push(bos);
            }
        }

        // Basic whitespace-delimited / exact match tokenizer for baseline scaffolding.
        // Full BPE / merges extraction wired in US-01.4.1 (#41).
        for word in text.split_whitespace() {
            if let Some(&id) = self.token_to_id.get(word) {
                tokens.push(id);
            } else {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "tokenizer".to_owned(),
                    reason: format!("unknown token '{word}' in input text"),
                });
            }
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, K3Error> {
        let mut result = String::new();
        for &id in tokens {
            if let Some(piece) = self.decode_piece(id)? {
                if !result.is_empty() {
                    result.push(' ');
                }
                result.push_str(&piece);
            }
        }
        Ok(result)
    }

    fn decode_piece(&self, token_id: u32) -> Result<Option<String>, K3Error> {
        if self.special_tokens.is_stop(token_id) {
            return Ok(None);
        }
        if let Some(token) = self.id_to_token.get(&token_id) {
            Ok(Some(token.clone()))
        } else {
            Err(K3Error::CapabilityMismatch {
                architecture: "tokenizer".to_owned(),
                reason: format!("unknown token ID {token_id} in vocabulary"),
            })
        }
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_encoding_and_decoding() {
        let mut tokenizer = SimpleTokenizer::new();
        tokenizer.set_special_tokens(SpecialTokens {
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            padding_token_id: Some(0),
            add_bos_token: true,
            stop_tokens: vec![2, 3],
        });

        tokenizer.add_token("hello", 10);
        tokenizer.add_token("world", 11);

        let encoded = tokenizer.encode("hello world").expect("encode succeeds");
        assert_eq!(encoded, vec![1, 10, 11]);

        let decoded = tokenizer.decode(&[10, 11]).expect("decode succeeds");
        assert_eq!(decoded, "hello world");

        // Streaming piece decode for stop token returns None
        assert_eq!(tokenizer.decode_piece(2).unwrap(), None);
        assert_eq!(
            tokenizer.decode_piece(10).unwrap(),
            Some("hello".to_owned())
        );

        // Unknown token returns CapabilityMismatch error
        assert!(matches!(
            tokenizer.encode("unregistered"),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }
}
