//! `k3-tokenizer`: Shared tokenizer, chat template, and streaming piece decoder.
//!
//! Provides decoupled tokenization interfaces shared between `k3-diag` and
//! `k3-manager` per ADR-025 without dragging in engine compute or hardware
//! affinity dependencies.

use std::collections::HashMap;

use k3_core::{GgufArray, GgufFile, GgufValue, K3Error};

/// GGUF token type classification (`tokenizer.ggml.token_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    /// Normal vocabulary piece (`1`).
    Normal,
    /// Unknown token (`2`).
    Unknown,
    /// Control / special token (`3`).
    Control,
    /// User-defined token (`4`).
    UserDefined,
    /// Unused token (`5`).
    Unused,
    /// Raw byte fallback token `<0xXX>` (`6`).
    Byte,
}

impl TokenType {
    /// Maps a GGUF `tokenizer.ggml.token_type` integer to [`TokenType`].
    #[must_use]
    pub const fn from_gguf_i32(val: i32) -> Self {
        match val {
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Normal,
        }
    }
}

/// Special control token IDs and stop sequences extracted from model metadata.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SpecialTokens {
    /// Beginning-of-sequence token ID (`tokenizer.ggml.bos_token_id`).
    pub bos_token_id: Option<u32>,
    /// End-of-sequence token ID (`tokenizer.ggml.eos_token_id`).
    pub eos_token_id: Option<u32>,
    /// End-of-turn token ID (`tokenizer.ggml.eot_token_id`).
    pub eot_token_id: Option<u32>,
    /// Padding token ID (`tokenizer.ggml.padding_token_id`).
    pub padding_token_id: Option<u32>,
    /// Whether the prompt should automatically prepend `bos_token_id`.
    pub add_bos_token: bool,
    /// Explicit stop sequence token IDs.
    pub stop_tokens: Vec<u32>,
    /// Text-level stop sequences (e.g., `"<|im_end|>"`, `"</s>"`).
    pub stop_sequences: Vec<String>,
}

impl SpecialTokens {
    /// Checks if a given token ID is considered a sequence stop token.
    #[must_use]
    pub fn is_stop(&self, token_id: u32) -> bool {
        if self.eos_token_id == Some(token_id) || self.eot_token_id == Some(token_id) {
            return true;
        }
        self.stop_tokens.contains(&token_id)
    }
}

/// Architecture-aware chat template formatter (ADR-023, ADR-025).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplate {
    raw_template: Option<String>,
}

impl ChatTemplate {
    /// Creates a new [`ChatTemplate`] from an optional GGUF `tokenizer.chat_template` string.
    #[must_use]
    pub const fn new(raw_template: Option<String>) -> Self {
        Self { raw_template }
    }

    /// Returns the raw template string from GGUF metadata, if present.
    #[must_use]
    pub fn raw_template(&self) -> Option<&str> {
        self.raw_template.as_deref()
    }

    /// Formats an operator user prompt according to the model's chat template rules.
    #[must_use]
    pub fn format_prompt(&self, user_prompt: &str) -> String {
        let Some(tmpl) = self.raw_template.as_deref() else {
            return user_prompt.to_owned();
        };

        if tmpl.contains("{{prompt}}") {
            return tmpl.replace("{{prompt}}", user_prompt);
        }
        if tmpl.contains("{{content}}") {
            return tmpl.replace("{{content}}", user_prompt);
        }
        if tmpl.contains("<|im_start|>") {
            return format!("<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n");
        }
        if tmpl.contains("[INST]") {
            return format!("[INST] {user_prompt} [/INST]");
        }

        user_prompt.to_owned()
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

/// Full GGUF-backed vocabulary, BPE/SentencePiece tokenizer, and chat template engine (US-01.4).
#[derive(Debug, Clone)]
pub struct GgufTokenizer {
    model_type: String,
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
    scores: Vec<f32>,
    token_types: Vec<TokenType>,
    merges: Vec<(String, String)>,
    merge_ranks: HashMap<(String, String), usize>,
    byte_tokens: [Option<u32>; 256],
    control_tokens: Vec<(String, u32)>,
    special_tokens: SpecialTokens,
    chat_template: ChatTemplate,
}

impl GgufTokenizer {
    /// Extracts vocabulary tokens, scores, token types, BPE merges, special control tokens,
    /// and chat template definitions from a parsed [`GgufFile`] (US-01.4.1, ADR-025).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if required vocabulary metadata (`tokenizer.ggml.tokens`)
    /// is missing or malformed.
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Result<Self, K3Error> {
        let arch = gguf.architecture();

        let model_type = gguf
            .metadata_value("tokenizer.ggml.model")
            .and_then(GgufValue::as_str)
            .unwrap_or("gpt2")
            .to_owned();

        let tokens_val = gguf
            .metadata_value("tokenizer.ggml.tokens")
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: "missing required 'tokenizer.ggml.tokens' array in GGUF metadata"
                    .to_owned(),
            })?;

        let id_to_token: Vec<String> = match tokens_val.as_array() {
            Some(GgufArray::String(items)) => items.iter().map(|s| (*s).to_owned()).collect(),
            _ => {
                return Err(K3Error::CapabilityMismatch {
                    architecture: arch.to_owned(),
                    reason: "'tokenizer.ggml.tokens' must be a string array".to_owned(),
                });
            }
        };

        if id_to_token.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: "'tokenizer.ggml.tokens' vocabulary array cannot be empty".to_owned(),
            });
        }

        let scores: Vec<f32> = match gguf
            .metadata_value("tokenizer.ggml.scores")
            .and_then(GgufValue::as_array)
        {
            Some(GgufArray::Float32(vals)) => vals.clone(),
            _ => vec![0.0_f32; id_to_token.len()],
        };

        let token_types: Vec<TokenType> = match gguf
            .metadata_value("tokenizer.ggml.token_type")
            .and_then(GgufValue::as_array)
        {
            Some(GgufArray::Int32(vals)) => {
                vals.iter().copied().map(TokenType::from_gguf_i32).collect()
            }
            Some(GgufArray::Uint32(vals)) => vals
                .iter()
                .copied()
                .map(|u| TokenType::from_gguf_i32(i32::try_from(u).unwrap_or(1)))
                .collect(),
            _ => vec![TokenType::Normal; id_to_token.len()],
        };

        let mut merges = Vec::new();
        let mut merge_ranks = HashMap::new();
        if let Some(GgufArray::String(raw_merges)) = gguf
            .metadata_value("tokenizer.ggml.merges")
            .and_then(GgufValue::as_array)
        {
            for (rank, entry) in raw_merges.iter().enumerate() {
                if let Some((left, right)) = entry.split_once(' ') {
                    let pair = (left.to_owned(), right.to_owned());
                    merge_ranks.entry(pair.clone()).or_insert(rank);
                    merges.push(pair);
                }
            }
        }

        let mut token_to_id = HashMap::with_capacity(id_to_token.len());
        let mut byte_tokens = [None; 256];
        let mut control_tokens = Vec::new();

        for (idx, text) in id_to_token.iter().enumerate() {
            let Ok(id) = u32::try_from(idx) else {
                continue;
            };
            token_to_id.entry(text.clone()).or_insert(id);

            if let Some(byte_val) = parse_hex_byte_token(text) {
                byte_tokens[usize::from(byte_val)] = Some(id);
            }

            let ttype = token_types.get(idx).copied().unwrap_or(TokenType::Normal);
            if ttype == TokenType::Control && !text.is_empty() {
                control_tokens.push((text.clone(), id));
            }
        }

        // Sort control tokens longest-first so greedy matching prefers longer markers
        control_tokens.sort_by_key(|b| std::cmp::Reverse(b.0.len()));

        let bos_token_id = gguf
            .metadata_value("tokenizer.ggml.bos_token_id")
            .and_then(GgufValue::as_u32);
        let eos_token_id = gguf
            .metadata_value("tokenizer.ggml.eos_token_id")
            .and_then(GgufValue::as_u32);
        let eot_token_id = gguf
            .metadata_value("tokenizer.ggml.eot_token_id")
            .and_then(GgufValue::as_u32);
        let padding_token_id = gguf
            .metadata_value("tokenizer.ggml.padding_token_id")
            .and_then(GgufValue::as_u32);
        let add_bos_token = gguf
            .metadata_value("tokenizer.ggml.add_bos_token")
            .and_then(GgufValue::as_bool)
            .unwrap_or(false);

        let mut stop_tokens = Vec::new();
        if let Some(eos) = eos_token_id {
            stop_tokens.push(eos);
        }
        if let Some(eot) = eot_token_id {
            if !stop_tokens.contains(&eot) {
                stop_tokens.push(eot);
            }
        }

        let mut stop_sequences = Vec::new();
        for marker in ["<|im_end|>", "<|eot_id|>", "</s>", "<|endoftext|>"] {
            if let Some(&id) = token_to_id.get(marker) {
                if !stop_tokens.contains(&id) {
                    stop_tokens.push(id);
                }
            }
            stop_sequences.push(marker.to_owned());
        }

        let special_tokens = SpecialTokens {
            bos_token_id,
            eos_token_id,
            eot_token_id,
            padding_token_id,
            add_bos_token,
            stop_tokens,
            stop_sequences,
        };

        let chat_template_str = gguf
            .metadata_value("tokenizer.chat_template")
            .and_then(GgufValue::as_str)
            .map(str::to_owned);
        let chat_template = ChatTemplate::new(chat_template_str);

        Ok(Self {
            model_type,
            id_to_token,
            token_to_id,
            scores,
            token_types,
            merges,
            merge_ranks,
            byte_tokens,
            control_tokens,
            special_tokens,
            chat_template,
        })
    }

    /// Returns the GGUF tokenizer model family (e.g. `"gpt2"`, `"llama"`).
    #[must_use]
    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    /// Returns the total vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Looks up a token ID by its exact vocabulary string.
    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Looks up a token string by its vocabulary ID.
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        usize::try_from(id)
            .ok()
            .and_then(|idx| self.id_to_token.get(idx).map(String::as_str))
    }

    /// Returns the token score array extracted from GGUF metadata.
    #[must_use]
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    /// Returns the token type classification array extracted from GGUF metadata.
    #[must_use]
    pub fn token_types(&self) -> &[TokenType] {
        &self.token_types
    }

    /// Returns the BPE merge pairs extracted from GGUF metadata.
    #[must_use]
    pub fn merges(&self) -> &[(String, String)] {
        &self.merges
    }

    /// Returns the chat template associated with this tokenizer.
    #[must_use]
    pub const fn chat_template(&self) -> &ChatTemplate {
        &self.chat_template
    }

    /// Formats a user prompt with the model's chat template (if `apply_chat_template` is `true`),
    /// tokenizes it into token IDs, and validates that `tokens.len() < n_ctx` (US-01.4.2, ADR-027).
    ///
    /// # Errors
    ///
    /// - Returns [`K3Error::ContextLengthExceeded`] if `n_ctx == 0` or `tokens.len() >= n_ctx`.
    /// - Returns [`K3Error::CapabilityMismatch`] if input text contains unencodable sequences.
    pub fn tokenize_prompt(
        &self,
        prompt: &str,
        apply_chat_template: bool,
        n_ctx: usize,
    ) -> Result<Vec<u32>, K3Error> {
        let formatted = if apply_chat_template {
            self.chat_template.format_prompt(prompt)
        } else {
            prompt.to_owned()
        };

        let tokens = self.encode(&formatted)?;
        if n_ctx == 0 || tokens.len() >= n_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: tokens.len(),
                max_context_tokens: n_ctx,
            });
        }

        Ok(tokens)
    }

    /// Creates a new [`StreamingDetokenizer`] bound to this tokenizer's vocabulary and stop rules.
    #[must_use]
    pub fn streaming_detokenizer(&self) -> StreamingDetokenizer<'_> {
        StreamingDetokenizer::new(self)
    }

    fn encode_segment(&self, segment: &str, out: &mut Vec<u32>) -> Result<(), K3Error> {
        if segment.is_empty() {
            return Ok(());
        }

        // Fast path: exact match in vocabulary
        if let Some(&id) = self.token_to_id.get(segment) {
            out.push(id);
            return Ok(());
        }

        // SentencePiece style space normalization if model_type is "llama" and vocab uses '▁'
        let normalized;
        let text_to_process = if self.model_type == "llama"
            && self.token_to_id.keys().any(|k| k.contains('\u{2581}'))
        {
            normalized = segment.replace(' ', "\u{2581}");
            normalized.as_str()
        } else {
            segment
        };

        // If BPE merges are configured, attempt BPE merge encoding on words/whitespace chunks
        if !self.merge_ranks.is_empty() && self.try_bpe_encode(text_to_process, out) {
            return Ok(());
        }

        // Greedy longest-prefix match in vocabulary with byte-token fallback
        let mut cursor = 0usize;
        let bytes = text_to_process.as_bytes();
        while cursor < bytes.len() {
            let remaining = &text_to_process[cursor..];
            let mut matched: Option<(usize, u32)> = None;

            // Find the longest prefix of `remaining` present in `token_to_id`
            for (char_idx, ch) in remaining.char_indices() {
                let end = char_idx + ch.len_utf8();
                let candidate = &remaining[..end];
                if let Some(&id) = self.token_to_id.get(candidate) {
                    matched = Some((end, id));
                }
            }

            if let Some((len, id)) = matched {
                out.push(id);
                cursor += len;
            } else {
                // Fallback to `<0xXX>` byte token if available
                let b = bytes[cursor];
                if let Some(byte_id) = self.byte_tokens[usize::from(b)] {
                    out.push(byte_id);
                    cursor += 1;
                } else {
                    let ch = remaining.chars().next().unwrap_or('\0');
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "tokenizer".to_owned(),
                        reason: format!(
                            "unable to encode character '{ch}' (byte 0x{b:02X}) in segment '{segment}'"
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    fn try_bpe_encode(&self, text: &str, out: &mut Vec<u32>) -> bool {
        let mut symbols: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        if symbols.is_empty() {
            return true;
        }

        loop {
            if symbols.len() < 2 {
                break;
            }

            let mut best_pair: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..(symbols.len() - 1) {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if best_pair.is_none_or(|(best_rank, _)| rank < best_rank) {
                        best_pair = Some((rank, i));
                    }
                }
            }

            let Some((_, idx)) = best_pair else {
                break;
            };

            let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
            symbols[idx] = merged;
            symbols.remove(idx + 1);
        }

        let mut staged = Vec::with_capacity(symbols.len());
        for sym in &symbols {
            if let Some(&id) = self.token_to_id.get(sym.as_str()) {
                staged.push(id);
            } else {
                return false;
            }
        }

        out.extend(staged);
        true
    }

    fn token_raw_bytes(&self, token_id: u32) -> Result<Option<Vec<u8>>, K3Error> {
        let idx = usize::try_from(token_id).map_err(|_| K3Error::CapabilityMismatch {
            architecture: "tokenizer".to_owned(),
            reason: format!("token ID {token_id} overflows usize"),
        })?;

        let piece = self
            .id_to_token
            .get(idx)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "tokenizer".to_owned(),
                reason: format!("unknown token ID {token_id} in vocabulary"),
            })?;

        let ttype = self
            .token_types
            .get(idx)
            .copied()
            .unwrap_or(TokenType::Normal);
        if ttype == TokenType::Control {
            return Ok(None);
        }

        if let Some(byte_val) = parse_hex_byte_token(piece) {
            return Ok(Some(vec![byte_val]));
        }

        if self.model_type == "llama" && piece.contains('\u{2581}') {
            return Ok(Some(piece.replace('\u{2581}', " ").into_bytes()));
        }

        Ok(Some(piece.as_bytes().to_vec()))
    }
}

impl Tokenizer for GgufTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, K3Error> {
        let mut tokens = Vec::new();
        if self.special_tokens.add_bos_token {
            if let Some(bos) = self.special_tokens.bos_token_id {
                tokens.push(bos);
            }
        }

        let mut cursor = 0usize;
        while cursor < text.len() {
            let remaining = &text[cursor..];

            // 1. Check if `remaining` starts with a control token
            if let Some((ctrl_str, ctrl_id)) = self
                .control_tokens
                .iter()
                .find(|(ctrl, _)| remaining.starts_with(ctrl.as_str()))
            {
                tokens.push(*ctrl_id);
                cursor += ctrl_str.len();
                continue;
            }

            // 2. Find the next control token occurrence so we can encode the normal span before it
            let next_ctrl_offset = self
                .control_tokens
                .iter()
                .filter_map(|(ctrl, _)| remaining.find(ctrl.as_str()))
                .min()
                .unwrap_or(remaining.len());

            let span = &remaining[..next_ctrl_offset];
            self.encode_segment(span, &mut tokens)?;
            cursor += next_ctrl_offset;
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, K3Error> {
        let mut detok = self.streaming_detokenizer();
        let mut out = String::new();
        for &id in tokens {
            match detok.push_token(id)? {
                DetokenizeStep::Emit(piece) => out.push_str(&piece),
                DetokenizeStep::Hold => {}
                DetokenizeStep::Stop(prefix) => {
                    out.push_str(&prefix);
                    break;
                }
            }
        }
        out.push_str(&detok.finish());
        Ok(out)
    }

    fn decode_piece(&self, token_id: u32) -> Result<Option<String>, K3Error> {
        if self.special_tokens.is_stop(token_id) {
            return Ok(None);
        }
        let Some(bytes) = self.token_raw_bytes(token_id)? else {
            return Ok(None);
        };
        Ok(std::str::from_utf8(&bytes).ok().map(str::to_owned))
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }
}

/// Output state of a single step in [`StreamingDetokenizer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetokenizeStep {
    /// Emits a complete, valid UTF-8 text fragment ready for immediate `stdout` flushing.
    Emit(String),
    /// Holds buffered bytes because a multi-byte UTF-8 sequence or potential stop sequence is incomplete.
    Hold,
    /// Generation reached a stop token or text stop sequence; emits any remaining text preceding the stop marker.
    Stop(String),
}

/// Stateful incremental token-to-UTF-8 streaming detokenizer with stop-sequence detection (US-01.4.3).
#[derive(Debug)]
pub struct StreamingDetokenizer<'a> {
    tokenizer: &'a GgufTokenizer,
    pending_utf8_bytes: Vec<u8>,
    pending_text: String,
    stopped: bool,
}

impl<'a> StreamingDetokenizer<'a> {
    /// Creates a new [`StreamingDetokenizer`] for `tokenizer`.
    #[must_use]
    pub const fn new(tokenizer: &'a GgufTokenizer) -> Self {
        Self {
            tokenizer,
            pending_utf8_bytes: Vec::new(),
            pending_text: String::new(),
            stopped: false,
        }
    }

    /// Returns `true` if a stop token or stop sequence has been triggered.
    #[must_use]
    pub const fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Feeds the next generated `token_id` into the incremental detokenizer.
    ///
    /// Buffers incomplete UTF-8 multi-byte sequences across token boundaries without emitting
    /// `U+FFFD` replacement characters, and halts cleanly on token or text stop sequences.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `token_id` is out of vocabulary bounds.
    pub fn push_token(&mut self, token_id: u32) -> Result<DetokenizeStep, K3Error> {
        if self.stopped {
            return Ok(DetokenizeStep::Stop(String::new()));
        }

        // 1. Check token-ID stop condition (EOS / EOT / explicit stop_tokens)
        if self.tokenizer.special_tokens().is_stop(token_id) {
            self.stopped = true;
            let flushed = core::mem::take(&mut self.pending_text);
            return Ok(DetokenizeStep::Stop(flushed));
        }

        // 2. Append raw token bytes to pending UTF-8 buffer
        if let Some(raw_bytes) = self.tokenizer.token_raw_bytes(token_id)? {
            self.pending_utf8_bytes.extend_from_slice(&raw_bytes);
        } else {
            return Ok(DetokenizeStep::Hold);
        }

        // 3. Extract complete UTF-8 characters from `pending_utf8_bytes`
        self.drain_valid_utf8();

        if self.pending_text.is_empty() {
            return Ok(DetokenizeStep::Hold);
        }

        // 4. Check text-level stop sequences and partial stop-sequence suffixes
        let stop_seqs = &self.tokenizer.special_tokens().stop_sequences;

        // Check if any full stop sequence appears in `pending_text`
        let mut earliest_stop: Option<usize> = None;
        for stop in stop_seqs {
            if !stop.is_empty() {
                if let Some(pos) = self.pending_text.find(stop.as_str()) {
                    if earliest_stop.is_none_or(|prev| pos < prev) {
                        earliest_stop = Some(pos);
                    }
                }
            }
        }

        if let Some(stop_pos) = earliest_stop {
            self.stopped = true;
            let prefix = self.pending_text[..stop_pos].to_owned();
            self.pending_text.clear();
            return Ok(DetokenizeStep::Stop(prefix));
        }

        // Check if the tail of `pending_text` is a partial prefix of any stop sequence
        let holdback_len = longest_stop_prefix_suffix(&self.pending_text, stop_seqs);
        if holdback_len == self.pending_text.len() {
            return Ok(DetokenizeStep::Hold);
        }

        let emit_end = self.pending_text.len() - holdback_len;
        let emitted = self.pending_text[..emit_end].to_owned();
        self.pending_text.drain(..emit_end);

        if emitted.is_empty() {
            Ok(DetokenizeStep::Hold)
        } else {
            Ok(DetokenizeStep::Emit(emitted))
        }
    }

    /// Flushes any remaining held-back text at the end of generation.
    #[must_use]
    pub fn finish(&mut self) -> String {
        self.drain_valid_utf8();
        core::mem::take(&mut self.pending_text)
    }

    fn drain_valid_utf8(&mut self) {
        loop {
            if self.pending_utf8_bytes.is_empty() {
                break;
            }
            match std::str::from_utf8(&self.pending_utf8_bytes) {
                Ok(valid) => {
                    self.pending_text.push_str(valid);
                    self.pending_utf8_bytes.clear();
                    break;
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    if valid_up_to > 0 {
                        if let Ok(valid_prefix) =
                            std::str::from_utf8(&self.pending_utf8_bytes[..valid_up_to])
                        {
                            self.pending_text.push_str(valid_prefix);
                        }
                        self.pending_utf8_bytes.drain(..valid_up_to);
                    }

                    if let Some(invalid_len) = err.error_len() {
                        // Skip genuinely invalid UTF-8 byte sequence without emitting U+FFFD
                        self.pending_utf8_bytes.drain(..invalid_len);
                    } else {
                        // Incomplete multi-byte UTF-8 sequence at end of buffer; wait for next token
                        break;
                    }
                }
            }
        }
    }
}

fn longest_stop_prefix_suffix(text: &str, stop_sequences: &[String]) -> usize {
    let mut max_holdback = 0usize;
    for stop in stop_sequences {
        if stop.is_empty() {
            continue;
        }
        // Check all proper non-empty prefixes of `stop`
        for (char_idx, _) in stop.char_indices().skip(1) {
            let prefix = &stop[..char_idx];
            if text.ends_with(prefix) && prefix.len() > max_holdback {
                max_holdback = prefix.len();
            }
        }
    }
    max_holdback
}

fn parse_hex_byte_token(token: &str) -> Option<u8> {
    let inner = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if inner.len() == 2 {
        u8::from_str_radix(inner, 16).ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k3_core::test_helpers::{create_synthetic_gguf_v3, create_synthetic_gguf_v3_with_kvs};

    #[test]
    fn test_tokenizer_encoding_and_decoding() {
        let mut tokenizer = SimpleTokenizer::new();
        tokenizer.set_special_tokens(SpecialTokens {
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            eot_token_id: None,
            padding_token_id: Some(0),
            add_bos_token: true,
            stop_tokens: vec![2, 3],
            stop_sequences: Vec::new(),
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

    #[test]
    fn test_gguf_tokenizer_extraction_and_chat_template_us01_4_1_and_4_2() {
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &[]);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic gguf");

        let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("extract gguf tokenizer");
        assert_eq!(tokenizer.model_type(), "gpt2");
        assert_eq!(tokenizer.vocab_size(), 16);
        assert_eq!(tokenizer.special_tokens().bos_token_id, Some(1));
        assert_eq!(tokenizer.special_tokens().eos_token_id, Some(2));
        assert_eq!(tokenizer.special_tokens().padding_token_id, Some(0));
        assert!(tokenizer.special_tokens().add_bos_token);
        assert_eq!(tokenizer.token_to_id("hello"), Some(9));
        assert_eq!(tokenizer.id_to_token(10), Some("world"));
        assert!(!tokenizer.merges().is_empty());

        // Tokenize with chat template applied and n_ctx = 64
        let prompt_tokens = tokenizer
            .tokenize_prompt("hello world", true, 64)
            .expect("tokenize prompt within n_ctx");
        // Starts with BOS (1), followed by `<|im_start|>` (3), `user` (5), `\n` (7), `hello` (9), ` ` (8), `world` (10), `<|im_end|>` (4), `\n` (7), `<|im_start|>` (3), `assistant` (6), `\n` (7)
        assert_eq!(prompt_tokens[0], 1);
        assert_eq!(prompt_tokens[1], 3);
        assert!(prompt_tokens.contains(&9));
        assert!(prompt_tokens.contains(&10));

        // Prompt exceeding n_ctx must fail with ContextLengthExceeded (ADR-027)
        let ctx_err = tokenizer
            .tokenize_prompt("hello world", true, 5)
            .expect_err("must reject prompt exceeding n_ctx=5");
        assert_eq!(ctx_err.diagnostic_code(), "context_length_exceeded");
        assert!(matches!(
            ctx_err,
            K3Error::ContextLengthExceeded {
                max_context_tokens: 5,
                ..
            }
        ));
    }

    #[test]
    fn test_bpe_merges_and_multibyte_utf8_streaming_detokenizer_us01_4_3() {
        // Build vocabulary with BPE parts and multi-byte UTF-8 byte tokens for '🦀' (0xF0, 0x9F, 0xA6, 0x80)
        let tokens = vec![
            "<pad>", "<bos>", "<eos>", "h", "e", "l", "o", "he", "ll", "hello", "<0xF0>", "<0x9F>",
            "<0xA6>", "<0x80>", "<", "|im_end|", ">",
        ];
        let token_types = vec![3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 1, 1, 1];
        let merges = vec!["h e", "l l", "he ll", "hell o"];
        let extra_kvs = [
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(GgufArray::String(tokens)),
            ),
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array(GgufArray::Int32(token_types)),
            ),
            (
                "tokenizer.ggml.merges",
                GgufValue::Array(GgufArray::String(merges)),
            ),
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(false)),
        ];

        let bytes = create_synthetic_gguf_v3_with_kvs("synthetic", 32, &extra_kvs, &[]);
        let gguf = GgufFile::parse(&bytes).expect("parse gguf");
        let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("load tokenizer");

        // 1. Verify BPE merge encoding of "hello"
        let encoded = tokenizer.encode("hello").expect("encode hello");
        assert_eq!(encoded, vec![9]);

        // 2. Verify multi-byte UTF-8 ('🦀') split across 4 byte tokens decodes cleanly without U+FFFD
        let mut stream = tokenizer.streaming_detokenizer();
        assert_eq!(
            stream.push_token(9).unwrap(),
            DetokenizeStep::Emit("hello".to_owned())
        );
        assert_eq!(stream.push_token(10).unwrap(), DetokenizeStep::Hold); // 0xF0
        assert_eq!(stream.push_token(11).unwrap(), DetokenizeStep::Hold); // 0x9F
        assert_eq!(stream.push_token(12).unwrap(), DetokenizeStep::Hold); // 0xA6
        assert_eq!(
            stream.push_token(13).unwrap(),
            DetokenizeStep::Emit("🦀".to_owned())
        ); // 0x80 completes 4-byte UTF-8 sequence!

        // 3. Verify text stop-sequence detection ("<|im_end|>" split across tokens 14, 15, 16)
        assert_eq!(stream.push_token(14).unwrap(), DetokenizeStep::Hold); // "<" held back as prefix of "<|im_end|>"
        assert_eq!(stream.push_token(15).unwrap(), DetokenizeStep::Hold); // "<|im_end|" held back
        assert_eq!(
            stream.push_token(16).unwrap(),
            DetokenizeStep::Stop(String::new())
        ); // "<|im_end|>" triggers Stop without emitting stop marker!
        assert!(stream.is_stopped());
    }
}
