//! When typing the opening character of one of the possible pairs defined below,
//! this module provides the functionality to insert the paired closing character.

use crate::{graphemes, movement::Direction, Range, Rope, Selection, Tendril, Transaction};
use bitflags::bitflags;
use std::collections::HashMap;

use smallvec::SmallVec;

// Heavily based on https://github.com/codemirror/closebrackets/
pub const DEFAULT_PAIRS: &[(char, char)] = &[
    ('(', ')'),
    ('{', '}'),
    ('[', ']'),
    ('\'', '\''),
    ('"', '"'),
    ('`', '`'),
];

bitflags! {
    /// Context mask for where auto-pairing is allowed.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct ContextMask: u8 {
        /// Auto-pair in regular code context
        const CODE    = 0b0000_0001;
        /// Auto-pair inside string literals
        const STRING  = 0b0000_0010;
        /// Auto-pair inside comments
        const COMMENT = 0b0000_0100;
        /// Auto-pair inside regex literals
        const REGEX   = 0b0000_1000;
        /// Auto-pair in all contexts
        const ALL     = Self::CODE.bits() | Self::STRING.bits() | Self::COMMENT.bits() | Self::REGEX.bits();
    }
}

/// Classification of bracket pair types for features like rainbow brackets, surround, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BracketKind {
    /// Parentheses, braces, square brackets: (), {}, []
    #[default]
    Bracket,
    /// Quotes: ', ", `, """, '''
    Quote,
    /// Template/markup delimiters: {% %}, <!-- -->, etc.
    Delimiter,
    /// HTML/XML tags (future use)
    Tag,
    /// User-defined custom pair
    Custom,
}

/// The syntactic context at a position in the document.
///
/// Used to determine whether auto-pairing should be allowed based on
/// the `allowed_contexts` field of a `BracketPair`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BracketContext {
    /// Regular code (not inside string, comment, or regex)
    #[default]
    Code,
    /// Inside a string literal
    String,
    /// Inside a comment
    Comment,
    /// Inside a regex literal
    Regex,
    /// Context could not be determined (treat as Code)
    Unknown,
}

impl BracketContext {
    /// Convert this context to the corresponding ContextMask flag.
    pub fn to_mask(self) -> ContextMask {
        match self {
            BracketContext::Code | BracketContext::Unknown => ContextMask::CODE,
            BracketContext::String => ContextMask::STRING,
            BracketContext::Comment => ContextMask::COMMENT,
            BracketContext::Regex => ContextMask::REGEX,
        }
    }
}

/// Represents a multi-character bracket pair configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BracketPair {
    /// What the user types to trigger pairing (usually == open)
    pub trigger: String,
    /// Inserted on the left
    pub open: String,
    /// Inserted on the right
    pub close: String,
    /// Classification for features (surround, highlighting, etc.)
    pub kind: BracketKind,
    /// Where auto-pairing is allowed (code, string, comment, regex)
    pub allowed_contexts: ContextMask,
    /// Whether this pair participates in surround commands
    pub surround: bool,
}

impl BracketPair {
    /// Create a new bracket pair with default settings.
    pub fn new(open: impl Into<String>, close: impl Into<String>) -> Self {
        let open = open.into();
        let trigger = open.clone();
        Self {
            trigger,
            open,
            close: close.into(),
            kind: BracketKind::Bracket,
            allowed_contexts: ContextMask::CODE,
            surround: true,
        }
    }

    /// Create a bracket pair with a specific kind.
    pub fn with_kind(mut self, kind: BracketKind) -> Self {
        self.kind = kind;
        self
    }

    /// Set the allowed contexts for this pair.
    pub fn with_contexts(mut self, contexts: ContextMask) -> Self {
        self.allowed_contexts = contexts;
        self
    }

    /// Set whether this pair participates in surround commands.
    pub fn with_surround(mut self, surround: bool) -> Self {
        self.surround = surround;
        self
    }

    /// Set a custom trigger (different from open).
    pub fn with_trigger(mut self, trigger: impl Into<String>) -> Self {
        self.trigger = trigger.into();
        self
    }

    /// Returns true if open == close (symmetric pair like quotes).
    pub fn same(&self) -> bool {
        self.open == self.close
    }

    /// Returns the first character of the trigger.
    pub fn trigger_first_char(&self) -> Option<char> {
        self.trigger.chars().next()
    }

    /// Returns the first character of the close string.
    pub fn close_first_char(&self) -> Option<char> {
        self.close.chars().next()
    }

    /// Check if this pair should auto-close at the given position.
    pub fn should_close(&self, doc: &Rope, range: &Range) -> bool {
        let mut should_close = Self::next_is_not_alpha(doc, range);

        if self.same() {
            should_close &= Self::prev_is_not_alpha(doc, range);
        }

        should_close
    }

    fn next_is_not_alpha(doc: &Rope, range: &Range) -> bool {
        let cursor = range.cursor(doc.slice(..));
        let next_char = doc.get_char(cursor);
        next_char.map(|c| !c.is_alphanumeric()).unwrap_or(true)
    }

    fn prev_is_not_alpha(doc: &Rope, range: &Range) -> bool {
        let cursor = range.cursor(doc.slice(..));
        let prev_char = prev_char(doc, cursor);
        prev_char.map(|c| !c.is_alphanumeric()).unwrap_or(true)
    }
}

impl From<(char, char)> for BracketPair {
    fn from((open, close): (char, char)) -> Self {
        let kind = if open == close {
            BracketKind::Quote
        } else {
            BracketKind::Bracket
        };
        BracketPair::new(open.to_string(), close.to_string()).with_kind(kind)
    }
}

impl From<&(char, char)> for BracketPair {
    fn from(&(open, close): &(char, char)) -> Self {
        (open, close).into()
    }
}

impl From<(&str, &str)> for BracketPair {
    fn from((open, close): (&str, &str)) -> Self {
        let kind = if open == close {
            BracketKind::Quote
        } else if open.len() > 1 || close.len() > 1 {
            BracketKind::Delimiter
        } else {
            BracketKind::Bracket
        };
        BracketPair::new(open, close).with_kind(kind)
    }
}

/// Fast-lookup container for bracket pairs.
///
/// Provides O(1) lookup by first trigger character, with support for
/// multi-character triggers through longest-match semantics.
#[derive(Debug, Clone, Default)]
pub struct BracketSet {
    /// All configured pairs
    pairs: Vec<BracketPair>,
    /// Map from first trigger char to indices in `pairs`
    first_char_index: HashMap<char, Vec<usize>>,
    /// Map from first close char to indices in `pairs` (for skip-over detection)
    close_char_index: HashMap<char, Vec<usize>>,
    /// Longest trigger length (for sliding window search)
    max_trigger_len: usize,
}

impl BracketSet {
    /// Create a new BracketSet from a list of pairs.
    pub fn new(pairs: Vec<BracketPair>) -> Self {
        let mut first_char_index: HashMap<char, Vec<usize>> = HashMap::new();
        let mut close_char_index: HashMap<char, Vec<usize>> = HashMap::new();
        let mut max_trigger_len = 0;

        for (i, pair) in pairs.iter().enumerate() {
            if let Some(ch) = pair.trigger_first_char() {
                first_char_index.entry(ch).or_default().push(i);
            }
            if let Some(ch) = pair.close_first_char() {
                close_char_index.entry(ch).or_default().push(i);
            }
            max_trigger_len = max_trigger_len.max(pair.trigger.len());
        }

        Self {
            pairs,
            first_char_index,
            close_char_index,
            max_trigger_len,
        }
    }

    /// Create a BracketSet from default single-char pairs.
    pub fn from_default_pairs() -> Self {
        let pairs: Vec<BracketPair> = DEFAULT_PAIRS.iter().map(|p| p.into()).collect();
        Self::new(pairs)
    }

    /// Get all pairs.
    pub fn pairs(&self) -> &[BracketPair] {
        &self.pairs
    }

    /// Get pairs whose trigger starts with the given character.
    pub fn candidates_for_trigger(&self, ch: char) -> impl Iterator<Item = &BracketPair> {
        self.first_char_index
            .get(&ch)
            .into_iter()
            .flatten()
            .map(|&i| &self.pairs[i])
    }

    /// Get pairs whose close starts with the given character.
    pub fn candidates_for_close(&self, ch: char) -> impl Iterator<Item = &BracketPair> {
        self.close_char_index
            .get(&ch)
            .into_iter()
            .flatten()
            .map(|&i| &self.pairs[i])
    }

    /// Get pairs that participate in surround commands.
    pub fn surround_pairs(&self) -> impl Iterator<Item = &BracketPair> {
        self.pairs.iter().filter(|p| p.surround)
    }

    /// Look up a surround pair by a single character.
    ///
    /// Searches surround-enabled pairs for one where the character matches
    /// either the first character of open or close. Returns the (open, close)
    /// strings if found.
    ///
    /// Prefers single-char pairs over multi-char pairs when the char matches
    /// the first character of multiple pairs.
    pub fn get_pair_by_char(&self, ch: char) -> Option<(&str, &str)> {
        self.surround_pairs()
            .filter(|p| {
                let open_first = p.open.chars().next();
                let close_first = p.close.chars().next();
                open_first == Some(ch) || close_first == Some(ch)
            })
            .min_by_key(|p| p.open.len())
            .map(|p| (p.open.as_str(), p.close.as_str()))
    }

    /// Get surround strings for a character, with fallback.
    ///
    /// If the character matches a known surround pair, returns (open, close).
    /// Otherwise, returns the character as both open and close (symmetric pair).
    ///
    /// This mirrors the behavior of `match_brackets::get_pair()` but works
    /// with the `BracketSet` configuration.
    pub fn get_surround_strings(&self, ch: char) -> (String, String) {
        match self.get_pair_by_char(ch) {
            Some((open, close)) => (open.to_string(), close.to_string()),
            None => (ch.to_string(), ch.to_string()),
        }
    }

    /// Get the maximum trigger length.
    pub fn max_trigger_len(&self) -> usize {
        self.max_trigger_len
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Get the number of pairs.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }
}

/// Detect the longest matching trigger at the cursor position.
///
/// This looks backward from the cursor to find completed multi-character triggers.
/// Returns the matching pair if found.
pub fn detect_trigger_at<'a>(
    doc: &Rope,
    cursor_char: usize,
    last_typed: char,
    set: &'a BracketSet,
) -> Option<&'a BracketPair> {
    // Fast filter by last char of trigger (which is what was just typed)
    let candidates: Vec<_> = set
        .pairs()
        .iter()
        .filter(|pair| pair.trigger.ends_with(last_typed))
        .collect();

    if candidates.is_empty() {
        return None;
    }

    // For single-char triggers, just return the first match
    if candidates.iter().all(|p| p.trigger.len() == 1) {
        return candidates.into_iter().next();
    }

    // Build sliding window of recent chars (up to max_trigger_len)
    let start = cursor_char.saturating_sub(set.max_trigger_len().saturating_sub(1));
    let slice = doc.slice(start..cursor_char);
    let recent: String = slice.chars().chain(std::iter::once(last_typed)).collect();

    // Find longest matching trigger (greedy)
    candidates
        .into_iter()
        .filter(|pair| recent.ends_with(&pair.trigger))
        .max_by_key(|pair| pair.trigger.len())
}

/// Find if the prefix of a multi-char trigger was already auto-paired with a
/// single-char close that needs to be replaced.
///
/// For example, if we have pairs `{` → `}` and `{%` → `%}`:
/// - User types `{`, we insert `{}`
/// - User types `%`, we detect `{%` trigger
/// - This function finds that `}` at cursor should be replaced
///
/// Returns the character position of the close char to remove, if any.
fn find_prefix_close_to_replace(
    doc: &Rope,
    cursor: usize,
    matched_pair: &BracketPair,
    set: &BracketSet,
) -> Option<usize> {
    if matched_pair.trigger.chars().count() <= 1 {
        return None;
    }

    // Get all trigger chars into a small stack buffer (triggers are short).
    let mut chars: SmallVec<[char; 8]> = matched_pair.trigger.chars().collect();
    // Remove the last char (the one we just typed).
    let _ = chars.pop()?;
    let prefix_last_char = *chars.last()?;

    // Single-char prefix pair
    let prefix_pair = set.pairs().iter().find(|p| {
        p.trigger.chars().count() == 1
            && p.trigger.chars().next() == Some(prefix_last_char)
            && !p.same()
    })?;

    let close_first_char = prefix_pair.close.chars().next()?;
    let char_at_cursor = doc.get_char(cursor)?;

    if char_at_cursor == close_first_char {
        Some(cursor)
    } else {
        None
    }
}

/// Check if a pair should be active in the given context.
///
/// Returns true if the pair's `allowed_contexts` mask intersects with
/// the mask corresponding to the given context.
pub fn context_allows_pair(context: BracketContext, pair: &BracketPair) -> bool {
    pair.allowed_contexts.intersects(context.to_mask())
}

/// Detect if a close sequence matches at the cursor position.
///
/// This looks forward from the cursor to see if the close sequence is present.
pub fn detect_close_at<'a>(
    doc: &Rope,
    cursor_char: usize,
    last_typed: char,
    set: &'a BracketSet,
) -> Option<&'a BracketPair> {
    // Get candidates whose close starts with the typed char
    let candidates: Vec<_> = set.candidates_for_close(last_typed).collect();

    if candidates.is_empty() {
        return None;
    }

    // Check which pairs have their close sequence at cursor
    for pair in candidates {
        let close_len = pair.close.chars().count();
        if cursor_char + close_len > doc.len_chars() {
            continue;
        }

        if doc.slice(cursor_char..cursor_char + close_len) == pair.close {
            return Some(pair);
        }
    }

    None
}

/// Result of detecting a bracketed pair around the cursor for deletion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeletePairResult {
    /// Number of characters to delete before the cursor (the open sequence)
    pub delete_before: usize,
    /// Number of characters to delete after the cursor (the close sequence)
    pub delete_after: usize,
}

/// Detect if the cursor is positioned between a matched open/close pair for deletion.
///
/// This function checks if the characters immediately before the cursor match
/// an opening sequence, and the characters immediately after match the corresponding
/// closing sequence. If so, returns how many characters to delete on each side.
///
/// For example, with pairs `{` → `}` and `{%` → `%}`:
/// - `{|}` returns `DeletePairResult { delete_before: 1, delete_after: 1 }`
/// - `{%|%}` returns `DeletePairResult { delete_before: 2, delete_after: 2 }`
///
/// Returns `None` if the cursor is not between a matched pair.
pub fn detect_pair_for_deletion(
    doc: &Rope,
    cursor: usize,
    set: &BracketSet,
) -> Option<DeletePairResult> {
    if cursor == 0 {
        return None;
    }

    let doc_len = doc.len_chars();
    let mut best_match: Option<DeletePairResult> = None;

    for pair in set.pairs() {
        let open_len = pair.open.chars().count();
        let close_len = pair.close.chars().count();

        // Check if there's enough space before cursor for the open sequence
        if cursor < open_len {
            continue;
        }

        // Check if there's enough space after cursor for the close sequence
        if cursor + close_len > doc_len {
            continue;
        }

        let open_start = cursor - open_len;

        // Check if both match
        if doc.slice(open_start..cursor) == pair.open
            && doc.slice(cursor..cursor + close_len) == pair.close
        {
            // Prefer longer matches (e.g., `{%|%}` over `{|}` when both could match)
            if best_match.as_ref().map_or(true, |m| {
                open_len + close_len > m.delete_before + m.delete_after
            }) {
                best_match = Some(DeletePairResult {
                    delete_before: open_len,
                    delete_after: close_len,
                });
            }
        }
    }

    best_match
}

/// The type that represents the collection of auto pairs,
/// keyed by both opener and closer.
#[deprecated(
    note = "Use BracketSet instead, which supports multi-character pairs and context awareness"
)]
#[derive(Debug, Clone)]
pub struct AutoPairs(HashMap<char, Pair>);

/// Represents the config for a particular pairing.
#[deprecated(
    note = "Use BracketPair instead, which supports multi-character pairs and context awareness"
)]
#[derive(Debug, Clone, Copy)]
pub struct Pair {
    pub open: char,
    pub close: char,
}

impl Pair {
    /// true if open == close
    pub fn same(&self) -> bool {
        self.open == self.close
    }

    /// true if all of the pair's conditions hold for the given document and range
    pub fn should_close(&self, doc: &Rope, range: &Range) -> bool {
        let mut should_close = Self::next_is_not_alpha(doc, range);

        if self.same() {
            should_close &= Self::prev_is_not_alpha(doc, range);
        }

        should_close
    }

    pub fn next_is_not_alpha(doc: &Rope, range: &Range) -> bool {
        let cursor = range.cursor(doc.slice(..));
        let next_char = doc.get_char(cursor);
        next_char.map(|c| !c.is_alphanumeric()).unwrap_or(true)
    }

    pub fn prev_is_not_alpha(doc: &Rope, range: &Range) -> bool {
        let cursor = range.cursor(doc.slice(..));
        let prev_char = prev_char(doc, cursor);
        prev_char.map(|c| !c.is_alphanumeric()).unwrap_or(true)
    }
}

impl From<&(char, char)> for Pair {
    fn from(&(open, close): &(char, char)) -> Self {
        Self { open, close }
    }
}

impl From<(&char, &char)> for Pair {
    fn from((open, close): (&char, &char)) -> Self {
        Self {
            open: *open,
            close: *close,
        }
    }
}

impl AutoPairs {
    /// Make a new AutoPairs set with the given pairs and default conditions.
    pub fn new<'a, V, A>(pairs: V) -> Self
    where
        V: IntoIterator<Item = A> + 'a,
        A: Into<Pair>,
    {
        let mut auto_pairs = HashMap::new();

        for pair in pairs.into_iter() {
            let auto_pair = pair.into();

            auto_pairs.insert(auto_pair.open, auto_pair);

            if auto_pair.open != auto_pair.close {
                auto_pairs.insert(auto_pair.close, auto_pair);
            }
        }

        Self(auto_pairs)
    }

    pub fn get(&self, ch: char) -> Option<&Pair> {
        self.0.get(&ch)
    }
}

impl Default for AutoPairs {
    fn default() -> Self {
        AutoPairs::new(DEFAULT_PAIRS.iter())
    }
}

impl From<&BracketSet> for AutoPairs {
    fn from(bracket_set: &BracketSet) -> Self {
        let pairs: Vec<(char, char)> = bracket_set
            .pairs()
            .iter()
            .filter(|p| p.open.len() == 1 && p.close.len() == 1)
            .filter_map(|p| {
                let open = p.open.chars().next()?;
                let close = p.close.chars().next()?;
                Some((open, close))
            })
            .collect();
        AutoPairs::new(pairs.iter())
    }
}

// insert hook:
// Fn(doc, selection, char) => Option<Transaction>
// problem is, we want to do this per range, so we can call default handler for some ranges
// so maybe ret Vec<Option<Change>>
// but we also need to be able to return transactions...
//
// to simplify, maybe return Option<Transaction> and just reimplement the default

// [TODO]
// * delete implementation where it erases the whole bracket (|) -> |
// * change to multi character pairs to handle cases like placing the cursor in the
//   middle of triple quotes, and more exotic pairs like Jinja's {% %}

#[must_use]
pub fn hook(doc: &Rope, selection: &Selection, ch: char, pairs: &AutoPairs) -> Option<Transaction> {
    log::trace!("autopairs hook selection: {:#?}", selection);

    if let Some(pair) = pairs.get(ch) {
        if pair.same() {
            return Some(handle_same(doc, selection, pair));
        } else if pair.open == ch {
            return Some(handle_open(doc, selection, pair));
        } else if pair.close == ch {
            // && char_at pos == close
            return Some(handle_close(doc, selection, pair));
        }
    }

    None
}

/// State passed to the auto-pairs hook for context-aware pairing.
///
/// This struct encapsulates all the information needed to determine whether
/// auto-pairing should occur and what the result should be.
#[derive(Debug, Clone)]
pub struct AutoPairState<'a> {
    /// The document being edited
    pub doc: &'a Rope,
    /// The current selection
    pub selection: &'a Selection,
    /// The bracket set to use for pairing
    pub pairs: &'a BracketSet,
    /// The syntactic context at each cursor position (indexed by selection range)
    /// If None, context checking is disabled and all pairs are allowed.
    pub contexts: Option<&'a [BracketContext]>,
}

impl<'a> AutoPairState<'a> {
    /// Create a new AutoPairState without context information.
    pub fn new(doc: &'a Rope, selection: &'a Selection, pairs: &'a BracketSet) -> Self {
        Self {
            doc,
            selection,
            pairs,
            contexts: None,
        }
    }

    /// Create a new AutoPairState with context information.
    pub fn with_contexts(
        doc: &'a Rope,
        selection: &'a Selection,
        pairs: &'a BracketSet,
        contexts: &'a [BracketContext],
    ) -> Self {
        Self {
            doc,
            selection,
            pairs,
            contexts: Some(contexts),
        }
    }

    /// Get the context for a specific range index, defaulting to Code if not available.
    fn context_for_range(&self, range_idx: usize) -> BracketContext {
        self.contexts
            .and_then(|ctx| ctx.get(range_idx).copied())
            .unwrap_or(BracketContext::Code)
    }
}

/// Hook for multi-character auto-pairs with context awareness.
///
/// This is the context-aware entry point that supports multi-character pairs like ```,
/// `{% %}`, `<!-- -->`, etc., and respects syntactic context (code, string, comment).
///
/// Returns `Some(Transaction)` when auto-pairing should occur. The transaction
/// includes BOTH the typed character AND the closing sequence.
/// Returns `None` when no auto-pair action is needed (caller should insert normally).
#[must_use]
pub fn hook_with_context(state: &AutoPairState<'_>, ch: char) -> Option<Transaction> {
    log::trace!(
        "autopairs hook_with_context selection: {:#?}",
        state.selection
    );

    let mut end_ranges = SmallVec::with_capacity(state.selection.len());
    let mut offs = 0;
    let mut made_changes = false;

    let transaction = Transaction::change_by_selection(state.doc, state.selection, |start_range| {
        let cursor = start_range.cursor(state.doc.slice(..));
        let range_idx = state
            .selection
            .ranges()
            .iter()
            .position(|r| r == start_range)
            .unwrap_or(0);
        let context = state.context_for_range(range_idx);

        // Try to detect a completed trigger (the char is about to be typed at cursor)
        if let Some(pair) = detect_trigger_at(state.doc, cursor, ch, state.pairs) {
            // Check context gating
            if !context_allows_pair(context, pair) {
                // Context doesn't allow this pair, insert just the typed char
                let mut t = Tendril::new();
                t.push(ch);
                let next_range = get_next_range(state.doc, start_range, offs, 1);
                end_ranges.push(next_range);
                offs += 1;
                made_changes = true;
                return (cursor, cursor, Some(t));
            }

            let next_char = state.doc.get_char(cursor);

            // For symmetric pairs (quotes), check if we should skip over existing close
            if pair.same() && next_char == Some(ch) {
                // Skip over the close - no change needed, just move cursor
                let next_range = get_next_range(state.doc, start_range, offs, 0);
                end_ranges.push(next_range);
                made_changes = true;
                return (cursor, cursor, None);
            }

            // Check if we should insert the pair
            if pair.should_close(state.doc, start_range) {
                // Check if a prefix of this trigger was already auto-paired with a
                // single-char close that we need to replace.
                let prefix_close_to_remove =
                    find_prefix_close_to_replace(state.doc, cursor, pair, state.pairs);

                let mut pair_str = Tendril::new();
                pair_str.push(ch);
                pair_str.push_str(&pair.close);

                let len_inserted = pair_str.chars().count();

                // Calculate range changes accounting for removed close char
                let (delete_start, delete_end) =
                    if let Some(close_char_pos) = prefix_close_to_remove {
                        // Delete from cursor to after the old close char
                        (cursor, close_char_pos + 1)
                    } else {
                        (cursor, cursor)
                    };

                let chars_removed = delete_end - delete_start;
                let net_inserted = len_inserted.saturating_sub(chars_removed);

                let next_range = get_next_range(state.doc, start_range, offs, net_inserted);
                end_ranges.push(next_range);
                offs = offs + len_inserted - chars_removed;
                made_changes = true;
                return (delete_start, delete_end, Some(pair_str));
            } else {
                // Conditions not met for auto-close, insert just the typed char
                let mut t = Tendril::new();
                t.push(ch);
                let next_range = get_next_range(state.doc, start_range, offs, 1);
                end_ranges.push(next_range);
                offs += 1;
                made_changes = true;
                return (cursor, cursor, Some(t));
            }
        }

        // Check if we're typing a close character and should skip over it
        if let Some(pair) = detect_close_at(state.doc, cursor, ch, state.pairs) {
            if !pair.same() {
                // Non-symmetric pair: skip over the close
                let next_range = get_next_range(state.doc, start_range, offs, 0);
                end_ranges.push(next_range);
                made_changes = true;
                return (cursor, cursor, None);
            }
        }

        // No auto-pair action, return no-op (caller will insert normally)
        let next_range = get_next_range(state.doc, start_range, offs, 0);
        end_ranges.push(next_range);
        (cursor, cursor, None)
    });

    if made_changes {
        Some(
            transaction.with_selection(Selection::new(end_ranges, state.selection.primary_index())),
        )
    } else {
        None
    }
}

/// Hook for multi-character auto-pairs with automatic context detection from syntax tree.
///
/// This is the highest-level entry point for context-aware auto-pairing. It automatically
/// computes the syntactic context (code, string, comment, regex) for each cursor position
/// using the provided syntax tree and language data.
///
/// # Arguments
/// * `doc` - The document text
/// * `selection` - Current selection with one or more cursors
/// * `ch` - The character being typed
/// * `pairs` - The bracket set to use for auto-pairing
/// * `syntax` - Optional syntax tree (if None, falls back to CODE context)
/// * `lang_data` - Language data containing `bracket_context_at` method
/// * `loader` - Syntax loader for language configuration
///
/// # Returns
/// `Some(Transaction)` when auto-pairing should occur, `None` otherwise.
#[must_use]
pub fn hook_with_syntax(
    doc: &Rope,
    selection: &Selection,
    ch: char,
    pairs: &BracketSet,
    syntax: Option<&crate::syntax::Syntax>,
    lang_data: &crate::syntax::LanguageData,
    loader: &crate::syntax::Loader,
) -> Option<Transaction> {
    log::trace!("autopairs hook_with_syntax selection: {:#?}", selection);

    let contexts: Vec<BracketContext> = selection
        .ranges()
        .iter()
        .map(|range| {
            let cursor = range.cursor(doc.slice(..));
            match syntax {
                Some(syn) => {
                    lang_data.bracket_context_at(syn.tree(), doc.slice(..), cursor, loader)
                }
                None => BracketContext::Code,
            }
        })
        .collect();

    let state = AutoPairState::with_contexts(doc, selection, pairs, &contexts);
    hook_with_context(&state, ch)
}

/// Hook for multi-character auto-pairs.
///
/// This is the new entry point that supports multi-character pairs like ```,
/// `{% %}`, `<!-- -->`, etc.
///
/// Returns `Some(Transaction)` when auto-pairing should occur. The transaction
/// includes BOTH the typed character AND the closing sequence.
/// Returns `None` when no auto-pair action is needed (caller should insert normally).
#[must_use]
pub fn hook_multi(
    doc: &Rope,
    selection: &Selection,
    ch: char,
    pairs: &BracketSet,
) -> Option<Transaction> {
    log::trace!("autopairs hook_multi selection: {:#?}", selection);

    let mut end_ranges = SmallVec::with_capacity(selection.len());
    let mut offs = 0;
    let mut made_changes = false;

    let transaction = Transaction::change_by_selection(doc, selection, |start_range| {
        let cursor = start_range.cursor(doc.slice(..));

        // Try to detect a completed trigger (the char is about to be typed at cursor)
        if let Some(pair) = detect_trigger_at(doc, cursor, ch, pairs) {
            let next_char = doc.get_char(cursor);

            // For symmetric pairs (quotes), check if we should skip over existing close
            if pair.same() && next_char == Some(ch) {
                // Skip over the close - no change needed, just move cursor
                let next_range = get_next_range(doc, start_range, offs, 0);
                end_ranges.push(next_range);
                made_changes = true;
                return (cursor, cursor, None);
            }

            // Check if we should insert the pair
            if pair.should_close(doc, start_range) {
                // Check if a prefix of this trigger was already auto-paired with a
                // single-char close that we need to replace. For example, if the user
                // types "{" and we inserted "{}", then they type "%" to complete "{%",
                // we need to replace the "}" with "%}".
                let prefix_close_to_remove = find_prefix_close_to_replace(doc, cursor, pair, pairs);

                let mut pair_str = Tendril::new();
                pair_str.push(ch);
                pair_str.push_str(&pair.close);

                let len_inserted = pair_str.chars().count();

                // Calculate range changes accounting for removed close char
                let (delete_start, delete_end) =
                    if let Some(close_char_pos) = prefix_close_to_remove {
                        // Delete from cursor to after the old close char
                        (cursor, close_char_pos + 1)
                    } else {
                        (cursor, cursor)
                    };

                let chars_removed = delete_end - delete_start;
                let net_inserted = len_inserted.saturating_sub(chars_removed);

                let next_range = get_next_range(doc, start_range, offs, net_inserted);
                end_ranges.push(next_range);
                offs = offs + len_inserted - chars_removed;
                made_changes = true;
                return (delete_start, delete_end, Some(pair_str));
            } else {
                // Conditions not met for auto-close, insert just the typed char
                let mut t = Tendril::new();
                t.push(ch);
                let next_range = get_next_range(doc, start_range, offs, 1);
                end_ranges.push(next_range);
                offs += 1;
                made_changes = true;
                return (cursor, cursor, Some(t));
            }
        }

        // Check if we're typing a close character and should skip over it
        if let Some(pair) = detect_close_at(doc, cursor, ch, pairs) {
            if !pair.same() {
                // Non-symmetric pair: skip over the close
                let next_range = get_next_range(doc, start_range, offs, 0);
                end_ranges.push(next_range);
                made_changes = true;
                return (cursor, cursor, None);
            }
        }

        // No auto-pair action, return no-op (caller will insert normally)
        let next_range = get_next_range(doc, start_range, offs, 0);
        end_ranges.push(next_range);
        (cursor, cursor, None)
    });

    if made_changes {
        Some(transaction.with_selection(Selection::new(end_ranges, selection.primary_index())))
    } else {
        None
    }
}

fn prev_char(doc: &Rope, pos: usize) -> Option<char> {
    if pos == 0 {
        return None;
    }

    doc.get_char(pos - 1)
}

/// calculate what the resulting range should be for an auto pair insertion
fn get_next_range(doc: &Rope, start_range: &Range, offset: usize, len_inserted: usize) -> Range {
    // When the character under the cursor changes due to complete pair
    // insertion, we must look backward a grapheme and then add the length
    // of the insertion to put the resulting cursor in the right place, e.g.
    //
    // foo[\r\n] - anchor: 3, head: 5
    // foo([)]\r\n - anchor: 4, head: 5
    //
    // foo[\r\n] - anchor: 3, head: 5
    // foo'[\r\n] - anchor: 4, head: 6
    //
    // foo([)]\r\n - anchor: 4, head: 5
    // foo()[\r\n] - anchor: 5, head: 7
    //
    // [foo]\r\n - anchor: 0, head: 3
    // [foo(])\r\n - anchor: 0, head: 5

    // inserting at the very end of the document after the last newline
    if start_range.head == doc.len_chars() && start_range.anchor == doc.len_chars() {
        return Range::new(
            start_range.anchor + offset + 1,
            start_range.head + offset + 1,
        );
    }

    let doc_slice = doc.slice(..);
    let single_grapheme = start_range.is_single_grapheme(doc_slice);

    // just skip over graphemes
    if len_inserted == 0 {
        let end_anchor = if single_grapheme {
            graphemes::next_grapheme_boundary(doc_slice, start_range.anchor) + offset

        // even for backward inserts with multiple grapheme selections,
        // we want the anchor to stay where it is so that the relative
        // selection does not change, e.g.:
        //
        // foo([) wor]d -> insert ) -> foo()[ wor]d
        } else {
            start_range.anchor + offset
        };

        return Range::new(
            end_anchor,
            graphemes::next_grapheme_boundary(doc_slice, start_range.head) + offset,
        );
    }

    // trivial case: only inserted a single-char opener, just move the selection
    if len_inserted == 1 {
        let end_anchor = if single_grapheme || start_range.direction() == Direction::Backward {
            start_range.anchor + offset + 1
        } else {
            start_range.anchor + offset
        };

        return Range::new(end_anchor, start_range.head + offset + 1);
    }

    // If the head = 0, then we must be in insert mode with a backward
    // cursor, which implies the head will just move
    let end_head = if start_range.head == 0 || start_range.direction() == Direction::Backward {
        start_range.head + offset + 1
    } else {
        // We must have a forward cursor, which means we must move to the
        // other end of the grapheme to get to where the new characters
        // are inserted, then move the head to where it should be
        let prev_bound = graphemes::prev_grapheme_boundary(doc_slice, start_range.head);
        log::trace!(
            "prev_bound: {}, offset: {}, len_inserted: {}",
            prev_bound,
            offset,
            len_inserted
        );
        prev_bound + offset + len_inserted
    };

    let end_anchor = match (start_range.len(), start_range.direction()) {
        // if we have a zero width cursor, it shifts to the same number
        (0, _) => end_head,

        // If we are inserting for a regular one-width cursor, the anchor
        // moves with the head. This is the fast path for ASCII.
        (1, Direction::Forward) => end_head - 1,
        (1, Direction::Backward) => end_head + 1,

        (_, Direction::Forward) => {
            if single_grapheme {
                graphemes::prev_grapheme_boundary(doc.slice(..), start_range.head) + 1

            // if we are appending, the anchor stays where it is; only offset
            // for multiple range insertions
            } else {
                start_range.anchor + offset
            }
        }

        (_, Direction::Backward) => {
            if single_grapheme {
                // if we're backward, then the head is at the first char
                // of the typed char, so we need to add the length of
                // the closing char
                graphemes::prev_grapheme_boundary(doc.slice(..), start_range.anchor)
                    + len_inserted
                    + offset
            } else {
                // when we are inserting in front of a selection, we need to move
                // the anchor over by however many characters were inserted overall
                start_range.anchor + offset + len_inserted
            }
        }
    };

    Range::new(end_anchor, end_head)
}

fn handle_open(doc: &Rope, selection: &Selection, pair: &Pair) -> Transaction {
    let mut end_ranges = SmallVec::with_capacity(selection.len());
    let mut offs = 0;

    let transaction = Transaction::change_by_selection(doc, selection, |start_range| {
        let cursor = start_range.cursor(doc.slice(..));
        let next_char = doc.get_char(cursor);
        let len_inserted;

        // Since auto pairs are currently limited to single chars, we're either
        // inserting exactly one or two chars. When arbitrary length pairs are
        // added, these will need to be changed.
        let change = match next_char {
            Some(_) if !pair.should_close(doc, start_range) => {
                len_inserted = 1;
                let mut tendril = Tendril::new();
                tendril.push(pair.open);
                (cursor, cursor, Some(tendril))
            }
            _ => {
                // insert open & close
                let pair_str = Tendril::from_iter([pair.open, pair.close]);
                len_inserted = 2;
                (cursor, cursor, Some(pair_str))
            }
        };

        let next_range = get_next_range(doc, start_range, offs, len_inserted);
        end_ranges.push(next_range);
        offs += len_inserted;

        change
    });

    let t = transaction.with_selection(Selection::new(end_ranges, selection.primary_index()));
    log::debug!("auto pair transaction: {:#?}", t);
    t
}

fn handle_close(doc: &Rope, selection: &Selection, pair: &Pair) -> Transaction {
    let mut end_ranges = SmallVec::with_capacity(selection.len());
    let mut offs = 0;

    let transaction = Transaction::change_by_selection(doc, selection, |start_range| {
        let cursor = start_range.cursor(doc.slice(..));
        let next_char = doc.get_char(cursor);
        let mut len_inserted = 0;

        let change = if next_char == Some(pair.close) {
            // return transaction that moves past close
            (cursor, cursor, None) // no-op
        } else {
            len_inserted = 1;
            let mut tendril = Tendril::new();
            tendril.push(pair.close);
            (cursor, cursor, Some(tendril))
        };

        let next_range = get_next_range(doc, start_range, offs, len_inserted);
        end_ranges.push(next_range);
        offs += len_inserted;

        change
    });

    let t = transaction.with_selection(Selection::new(end_ranges, selection.primary_index()));
    log::debug!("auto pair transaction: {:#?}", t);
    t
}

/// handle cases where open and close is the same, or in triples ("""docstring""")
fn handle_same(doc: &Rope, selection: &Selection, pair: &Pair) -> Transaction {
    let mut end_ranges = SmallVec::with_capacity(selection.len());

    let mut offs = 0;

    let transaction = Transaction::change_by_selection(doc, selection, |start_range| {
        let cursor = start_range.cursor(doc.slice(..));
        let mut len_inserted = 0;
        let next_char = doc.get_char(cursor);

        let change = if next_char == Some(pair.open) {
            //  return transaction that moves past close
            (cursor, cursor, None) // no-op
        } else {
            let mut pair_str = Tendril::new();
            pair_str.push(pair.open);

            // for equal pairs, don't insert both open and close if either
            // side has a non-pair char
            if pair.should_close(doc, start_range) {
                pair_str.push(pair.close);
            }

            len_inserted += pair_str.chars().count();
            (cursor, cursor, Some(pair_str))
        };

        let next_range = get_next_range(doc, start_range, offs, len_inserted);
        end_ranges.push(next_range);
        offs += len_inserted;

        change
    });

    let t = transaction.with_selection(Selection::new(end_ranges, selection.primary_index()));
    log::debug!("auto pair transaction: {:#?}", t);
    t
}

/// Registry of auto-pairs loaded from auto-pairs.toml.
///
/// This provides a central store of per-language bracket configurations that
/// can be looked up by language name. Languages without explicit configuration
/// fall back to the `default` entry.
#[derive(Debug, Clone, Default)]
pub struct AutoPairsRegistry {
    languages: HashMap<String, BracketSet>,
    default: BracketSet,
}

impl AutoPairsRegistry {
    /// Create a new empty registry with default pairs.
    pub fn new() -> Self {
        Self {
            languages: HashMap::new(),
            default: BracketSet::from_default_pairs(),
        }
    }

    /// Load registry from a parsed TOML value (from auto-pairs.toml).
    ///
    /// The TOML structure is expected to be:
    /// ```toml
    /// [default]
    /// pairs = [{ open = "(", close = ")" }, ...]
    ///
    /// [rust]
    /// pairs = [{ open = "(", close = ")" }, ...]
    /// ```
    pub fn from_toml(value: &toml::Value) -> Result<Self, AutoPairsRegistryError> {
        let table = value
            .as_table()
            .ok_or(AutoPairsRegistryError::InvalidFormat("expected table"))?;

        let mut languages = HashMap::new();
        let mut default = BracketSet::from_default_pairs();

        for (key, val) in table {
            let pairs = Self::parse_pairs(val)?;
            let bracket_set = BracketSet::new(pairs);

            if key == "default" {
                default = bracket_set;
            } else {
                languages.insert(key.clone(), bracket_set);
            }
        }

        Ok(Self { languages, default })
    }

    fn parse_pairs(val: &toml::Value) -> Result<Vec<BracketPair>, AutoPairsRegistryError> {
        let pairs_val = val
            .get("pairs")
            .ok_or(AutoPairsRegistryError::InvalidFormat("missing 'pairs' key"))?;

        let pairs_arr = pairs_val
            .as_array()
            .ok_or(AutoPairsRegistryError::InvalidFormat(
                "'pairs' must be array",
            ))?;

        let mut pairs = Vec::with_capacity(pairs_arr.len());

        for pair_val in pairs_arr {
            let open = pair_val
                .get("open")
                .and_then(|v| v.as_str())
                .ok_or(AutoPairsRegistryError::InvalidFormat("pair missing 'open'"))?;

            let close = pair_val.get("close").and_then(|v| v.as_str()).ok_or(
                AutoPairsRegistryError::InvalidFormat("pair missing 'close'"),
            )?;

            let mut bracket_pair = BracketPair::new(open, close);

            if let Some(kind_str) = pair_val.get("kind").and_then(|v| v.as_str()) {
                bracket_pair = bracket_pair.with_kind(match kind_str {
                    "bracket" => BracketKind::Bracket,
                    "quote" => BracketKind::Quote,
                    "delimiter" => BracketKind::Delimiter,
                    "tag" => BracketKind::Tag,
                    "custom" => BracketKind::Custom,
                    _ => BracketKind::Bracket,
                });
            } else {
                let kind = if open == close {
                    BracketKind::Quote
                } else if open.len() > 1 || close.len() > 1 {
                    BracketKind::Delimiter
                } else {
                    BracketKind::Bracket
                };
                bracket_pair = bracket_pair.with_kind(kind);
            }

            if let Some(trigger) = pair_val.get("trigger").and_then(|v| v.as_str()) {
                bracket_pair = bracket_pair.with_trigger(trigger);
            }

            if let Some(surround) = pair_val.get("surround").and_then(|v| v.as_bool()) {
                bracket_pair = bracket_pair.with_surround(surround);
            }

            if let Some(contexts) = pair_val.get("allowed-contexts").and_then(|v| v.as_array()) {
                let mut mask = ContextMask::empty();
                for ctx in contexts {
                    if let Some(ctx_str) = ctx.as_str() {
                        match ctx_str {
                            "code" => mask |= ContextMask::CODE,
                            "string" => mask |= ContextMask::STRING,
                            "comment" => mask |= ContextMask::COMMENT,
                            "regex" => mask |= ContextMask::REGEX,
                            "all" => mask |= ContextMask::ALL,
                            _ => {}
                        }
                    }
                }
                if !mask.is_empty() {
                    bracket_pair = bracket_pair.with_contexts(mask);
                }
            }

            pairs.push(bracket_pair);
        }

        Ok(pairs)
    }

    /// Get the BracketSet for a language, falling back to default if not found.
    pub fn get(&self, language_name: &str) -> &BracketSet {
        self.languages.get(language_name).unwrap_or(&self.default)
    }

    /// Check if a specific language has configuration.
    pub fn has_language(&self, language_name: &str) -> bool {
        self.languages.contains_key(language_name)
    }

    /// Get the default BracketSet.
    pub fn default_set(&self) -> &BracketSet {
        &self.default
    }

    /// Get all configured language names.
    pub fn language_names(&self) -> impl Iterator<Item = &str> {
        self.languages.keys().map(|s| s.as_str())
    }
}

/// Error type for AutoPairsRegistry parsing.
#[derive(Debug, Clone)]
pub enum AutoPairsRegistryError {
    InvalidFormat(&'static str),
}

impl std::fmt::Display for AutoPairsRegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AutoPairsRegistryError::InvalidFormat(msg) => {
                write!(f, "invalid auto-pairs.toml format: {}", msg)
            }
        }
    }
}

impl std::error::Error for AutoPairsRegistryError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bracket_pair_creation() {
        let pair = BracketPair::new("(", ")");
        assert_eq!(pair.open, "(");
        assert_eq!(pair.close, ")");
        assert_eq!(pair.trigger, "(");
        assert!(!pair.same());
    }

    #[test]
    fn test_bracket_pair_same() {
        let pair = BracketPair::new("\"", "\"");
        assert!(pair.same());

        let pair = BracketPair::new("(", ")");
        assert!(!pair.same());
    }

    #[test]
    fn test_bracket_set_candidates() {
        let pairs = vec![
            BracketPair::new("(", ")"),
            BracketPair::new("{", "}"),
            BracketPair::new("{{", "}}"),
        ];
        let set = BracketSet::new(pairs);

        let candidates: Vec<_> = set.candidates_for_trigger('{').collect();
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_bracket_set_max_trigger_len() {
        let pairs = vec![
            BracketPair::new("(", ")"),
            BracketPair::new("```", "```"),
            BracketPair::new("<!--", "-->"),
        ];
        let set = BracketSet::new(pairs);

        assert_eq!(set.max_trigger_len(), 4);
    }

    #[test]
    fn test_context_mask() {
        let mask = ContextMask::CODE | ContextMask::STRING;
        assert!(mask.contains(ContextMask::CODE));
        assert!(mask.contains(ContextMask::STRING));
        assert!(!mask.contains(ContextMask::COMMENT));
    }

    #[test]
    fn test_bracket_kind() {
        let bracket = BracketPair::from(('(', ')'));
        assert_eq!(bracket.kind, BracketKind::Bracket);

        let quote = BracketPair::from(('"', '"'));
        assert_eq!(quote.kind, BracketKind::Quote);

        let delimiter = BracketPair::from(("<!--", "-->"));
        assert_eq!(delimiter.kind, BracketKind::Delimiter);
    }

    #[test]
    fn test_bracket_context_to_mask() {
        assert_eq!(BracketContext::Code.to_mask(), ContextMask::CODE);
        assert_eq!(BracketContext::String.to_mask(), ContextMask::STRING);
        assert_eq!(BracketContext::Comment.to_mask(), ContextMask::COMMENT);
        assert_eq!(BracketContext::Regex.to_mask(), ContextMask::REGEX);
        assert_eq!(BracketContext::Unknown.to_mask(), ContextMask::CODE);
    }

    #[test]
    fn test_bracket_context_allows_pair() {
        let bracket = BracketPair::new("(", ")").with_contexts(ContextMask::CODE);
        let quote =
            BracketPair::new("\"", "\"").with_contexts(ContextMask::CODE | ContextMask::STRING);

        // Bracket only allowed in code
        assert!(bracket
            .allowed_contexts
            .intersects(BracketContext::Code.to_mask()));
        assert!(!bracket
            .allowed_contexts
            .intersects(BracketContext::String.to_mask()));
        assert!(!bracket
            .allowed_contexts
            .intersects(BracketContext::Comment.to_mask()));

        // Quote allowed in code and string
        assert!(quote
            .allowed_contexts
            .intersects(BracketContext::Code.to_mask()));
        assert!(quote
            .allowed_contexts
            .intersects(BracketContext::String.to_mask()));
        assert!(!quote
            .allowed_contexts
            .intersects(BracketContext::Comment.to_mask()));
    }

    #[test]
    fn test_detect_trigger_single_char() {
        let doc = Rope::from("test");
        let pairs = vec![BracketPair::new("(", ")")];
        let set = BracketSet::new(pairs);

        let result = detect_trigger_at(&doc, 4, '(', &set);
        assert!(result.is_some());
        assert_eq!(result.unwrap().open, "(");
    }

    #[test]
    fn test_detect_trigger_multi_char() {
        let doc = Rope::from("test`");
        let pairs = vec![BracketPair::new("`", "`"), BracketPair::new("```", "```")];
        let set = BracketSet::new(pairs);

        // After typing single backtick
        let result = detect_trigger_at(&doc, 5, '`', &set);
        assert!(result.is_some());
        // Should match single backtick since that's what's in the doc
        assert_eq!(result.unwrap().open, "`");

        // Now test with triple backticks
        let doc = Rope::from("test``");
        let result = detect_trigger_at(&doc, 6, '`', &set);
        assert!(result.is_some());
        // Should match triple backtick (longest match)
        assert_eq!(result.unwrap().open, "```");
    }

    #[test]
    fn test_bracket_set_from_default() {
        let set = BracketSet::from_default_pairs();
        assert_eq!(set.len(), DEFAULT_PAIRS.len());
    }

    #[test]
    fn test_multi_char_pair_builder() {
        let pair = BracketPair::new("{%", "%}")
            .with_kind(BracketKind::Delimiter)
            .with_contexts(ContextMask::CODE)
            .with_surround(false);

        assert_eq!(pair.open, "{%");
        assert_eq!(pair.close, "%}");
        assert_eq!(pair.kind, BracketKind::Delimiter);
        assert_eq!(pair.allowed_contexts, ContextMask::CODE);
        assert!(!pair.surround);
    }

    #[test]
    fn test_hook_multi_single_char_insert() {
        // The hook is called BEFORE the character is inserted.
        // It inserts BOTH the typed char AND the closing char.
        let doc = Rope::from("test\n");
        let selection = Selection::single(4, 5); // cursor at end of "test", before \n
        let set = BracketSet::from_default_pairs();

        let result = hook_multi(&doc, &selection, '(', &set);
        assert!(result.is_some());

        let transaction = result.unwrap();
        let mut new_doc = doc.clone();
        assert!(transaction.apply(&mut new_doc));

        // hook_multi inserts "(" + ")" at cursor position
        assert_eq!(new_doc.to_string(), "test()\n");
    }

    #[test]
    fn test_hook_multi_triple_backtick() {
        // When we type the third backtick, the document already has ``
        // and cursor is at position 6 (after the two backticks).
        // The hook should detect ``` trigger and insert ` + ```
        let doc = Rope::from("test``\n");
        let selection = Selection::single(6, 7); // cursor after `` (on the \n)

        let pairs = vec![BracketPair::new("`", "`"), BracketPair::new("```", "```")];
        let set = BracketSet::new(pairs);

        let result = hook_multi(&doc, &selection, '`', &set);
        assert!(result.is_some());

        let transaction = result.unwrap();
        let mut new_doc = doc.clone();
        assert!(transaction.apply(&mut new_doc));

        // The hook inserts "`" + "```" at cursor position 6
        // So: "test``" + "`" + "```" + "\n" = "test``````\n"
        assert_eq!(new_doc.to_string(), "test``````\n");
    }

    #[test]
    fn test_hook_multi_jinja_delimiter() {
        // Document has "{" and we're typing "%"
        // The trigger is "{%" so after typing %, we should get "%" + "%}"
        let doc = Rope::from("test{\n");
        let selection = Selection::single(5, 6); // cursor after { (on the \n)

        let pairs = vec![BracketPair::new("{", "}"), BracketPair::new("{%", "%}")];
        let set = BracketSet::new(pairs);

        let result = hook_multi(&doc, &selection, '%', &set);
        assert!(result.is_some());

        let transaction = result.unwrap();
        let mut new_doc = doc.clone();
        assert!(transaction.apply(&mut new_doc));

        // The hook inserts "%" + "%}" at cursor position 5
        // So: "test{" + "%" + "%}" + "\n" = "test{%%}\n"
        assert_eq!(new_doc.to_string(), "test{%%}\n");
    }

    #[test]
    fn test_hook_multi_skip_over_close() {
        // Test skipping over existing close bracket
        let doc = Rope::from("test()\n");
        let selection = Selection::single(5, 6); // cursor between ( and )

        let set = BracketSet::from_default_pairs();

        let result = hook_multi(&doc, &selection, ')', &set);
        assert!(result.is_some());

        let transaction = result.unwrap();
        let mut new_doc = doc.clone();
        assert!(transaction.apply(&mut new_doc));

        // Document should be unchanged (just cursor movement)
        assert_eq!(new_doc.to_string(), "test()\n");
    }

    #[test]
    fn test_hook_multi_symmetric_quote_not_after_alpha() {
        // Quotes shouldn't auto-pair after alphanumeric - just insert the quote
        let doc = Rope::from("test\n");
        let selection = Selection::single(4, 5); // cursor right after "test"

        let set = BracketSet::from_default_pairs();

        // After an alphanumeric, quotes should NOT auto-pair, just insert single quote
        let result = hook_multi(&doc, &selection, '"', &set);
        assert!(result.is_some());

        let transaction = result.unwrap();
        let mut new_doc = doc.clone();
        assert!(transaction.apply(&mut new_doc));

        // Only the typed quote is inserted, not a pair
        assert_eq!(new_doc.to_string(), "test\"\n");
    }

    #[test]
    fn test_hook_multi_symmetric_quote_after_space() {
        // Quotes should auto-pair after space
        let doc = Rope::from("test \n");
        let selection = Selection::single(5, 6); // cursor after space

        let set = BracketSet::from_default_pairs();

        let result = hook_multi(&doc, &selection, '"', &set);
        assert!(result.is_some());

        let transaction = result.unwrap();
        let mut new_doc = doc.clone();
        assert!(transaction.apply(&mut new_doc));

        // Should have inserted " + "
        assert_eq!(new_doc.to_string(), "test \"\"\n");
    }

    #[test]
    fn test_hook_multi_no_match() {
        let doc = Rope::from("test\n");
        let selection = Selection::single(4, 5);

        let set = BracketSet::from_default_pairs();

        // 'x' is not a trigger for any pair
        let result = hook_multi(&doc, &selection, 'x', &set);
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_trigger_considers_position() {
        // When cursor is at position 5 and we type %,
        // we need to look at chars before position 5
        let doc = Rope::from("test{");
        let set = BracketSet::new(vec![
            BracketPair::new("{", "}"),
            BracketPair::new("{%", "%}"),
        ]);

        // Cursor is at 5, we're typing '%'
        // Characters before cursor are "test{"
        // The trigger "{%" should match "{" + "%" (last typed)
        let result = detect_trigger_at(&doc, 5, '%', &set);
        assert!(result.is_some());
        assert_eq!(result.unwrap().open, "{%");
    }

    #[test]
    fn test_hook_multi_replaces_prefix_close() {
        // Scenario: User has both { → } and {% → %} pairs configured.
        // They type "{" which auto-pairs to "{|}" (cursor at |).
        // Then they type "%" - we should replace "}" with "%}" to get "{%|%}"
        let doc = Rope::from("test{}\n");
        let selection = Selection::single(5, 6); // cursor between { and }

        let pairs = vec![BracketPair::new("{", "}"), BracketPair::new("{%", "%}")];
        let set = BracketSet::new(pairs);

        let result = hook_multi(&doc, &selection, '%', &set);
        assert!(result.is_some());

        let transaction = result.unwrap();
        let mut new_doc = doc.clone();
        assert!(transaction.apply(&mut new_doc));

        // The } should be replaced with %%}
        // Result: "test{" + "%" + "%}" + "\n" = "test{%%}\n"
        // Wait, that's wrong. Let me reconsider.
        //
        // Original: "test{}\n" with cursor at position 5 (between { and })
        // We type "%"
        // The transaction should:
        // - Delete from cursor (5) to after "}" (6)
        // - Insert "%%}" at position 5
        // Result: "test{%%}\n"
        assert_eq!(new_doc.to_string(), "test{%%}\n");
    }

    #[test]
    fn test_detect_pair_for_deletion_single_char() {
        let doc = Rope::from("test{}\n");
        let set = BracketSet::from_default_pairs();

        // Cursor at position 5 (between { and })
        let result = detect_pair_for_deletion(&doc, 5, &set);
        assert!(result.is_some());
        let del = result.unwrap();
        assert_eq!(del.delete_before, 1);
        assert_eq!(del.delete_after, 1);
    }

    #[test]
    fn test_detect_pair_for_deletion_multi_char() {
        // String: "test{%%}\n"
        // Positions: t(0) e(1) s(2) t(3) {(4) %(5) %(6) }(7) \n(8)
        // Cursor at position 6 means we're between "{%" and "%}"
        let doc = Rope::from("test{%%}\n");
        let pairs = vec![BracketPair::new("{", "}"), BracketPair::new("{%", "%}")];
        let set = BracketSet::new(pairs);

        // Cursor at position 6 (between {% and %})
        let result = detect_pair_for_deletion(&doc, 6, &set);
        assert!(result.is_some());
        let del = result.unwrap();
        assert_eq!(del.delete_before, 2);
        assert_eq!(del.delete_after, 2);
    }

    #[test]
    fn test_detect_pair_for_deletion_prefers_longer_match() {
        // When both { and {% could match, prefer the longer one
        let doc = Rope::from("{%%}\n");
        let pairs = vec![BracketPair::new("{", "}"), BracketPair::new("{%", "%}")];
        let set = BracketSet::new(pairs);

        // Cursor at position 2 (between {% and %})
        let result = detect_pair_for_deletion(&doc, 2, &set);
        assert!(result.is_some());
        let del = result.unwrap();
        assert_eq!(del.delete_before, 2);
        assert_eq!(del.delete_after, 2);
    }

    #[test]
    fn test_detect_pair_for_deletion_no_match() {
        let doc = Rope::from("test\n");
        let set = BracketSet::from_default_pairs();

        // Cursor at position 2, no brackets around
        let result = detect_pair_for_deletion(&doc, 2, &set);
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_pair_for_deletion_at_start() {
        let doc = Rope::from("{}\n");
        let set = BracketSet::from_default_pairs();

        // Cursor at position 0, can't delete before
        let result = detect_pair_for_deletion(&doc, 0, &set);
        assert!(result.is_none());
    }

    #[test]
    fn test_context_allows_pair_code() {
        let pair = BracketPair::new("(", ")").with_contexts(ContextMask::CODE);

        // CODE context should allow CODE-only pair
        assert!(context_allows_pair(BracketContext::Code, &pair));
        // STRING context should NOT allow CODE-only pair
        assert!(!context_allows_pair(BracketContext::String, &pair));
        // COMMENT context should NOT allow CODE-only pair
        assert!(!context_allows_pair(BracketContext::Comment, &pair));
        // Unknown falls back to CODE
        assert!(context_allows_pair(BracketContext::Unknown, &pair));
    }

    #[test]
    fn test_context_allows_pair_multi_context() {
        let pair =
            BracketPair::new("\"", "\"").with_contexts(ContextMask::CODE | ContextMask::STRING);

        assert!(context_allows_pair(BracketContext::Code, &pair));
        assert!(context_allows_pair(BracketContext::String, &pair));
        assert!(!context_allows_pair(BracketContext::Comment, &pair));
    }

    #[test]
    fn test_context_allows_pair_all_contexts() {
        let pair = BracketPair::new("(", ")").with_contexts(ContextMask::ALL);

        assert!(context_allows_pair(BracketContext::Code, &pair));
        assert!(context_allows_pair(BracketContext::String, &pair));
        assert!(context_allows_pair(BracketContext::Comment, &pair));
        assert!(context_allows_pair(BracketContext::Regex, &pair));
    }

    #[test]
    fn test_auto_pair_state_creation() {
        let doc = Rope::from("test\n");
        let selection = Selection::single(4, 5);
        let pairs = BracketSet::from_default_pairs();

        let state = AutoPairState::new(&doc, &selection, &pairs);
        assert!(state.contexts.is_none());

        let contexts = vec![BracketContext::Code];
        let state_with_ctx = AutoPairState::with_contexts(&doc, &selection, &pairs, &contexts);
        assert!(state_with_ctx.contexts.is_some());
    }

    #[test]
    fn test_hook_with_context_in_code() {
        let doc = Rope::from("test\n");
        let selection = Selection::single(4, 5);
        let pairs = BracketSet::from_default_pairs();
        let contexts = vec![BracketContext::Code];

        let state = AutoPairState::with_contexts(&doc, &selection, &pairs, &contexts);
        let result = hook_with_context(&state, '(');

        assert!(result.is_some());
        let transaction = result.unwrap();
        let mut new_doc = doc.clone();
        assert!(transaction.apply(&mut new_doc));
        assert_eq!(new_doc.to_string(), "test()\n");
    }

    #[test]
    fn test_hook_with_context_blocked_in_string() {
        let doc = Rope::from("test\n");
        let selection = Selection::single(4, 5);
        // Bracket only allowed in CODE context
        let pairs = BracketSet::new(vec![
            BracketPair::new("(", ")").with_contexts(ContextMask::CODE)
        ]);
        // But we're in a STRING context
        let contexts = vec![BracketContext::String];

        let state = AutoPairState::with_contexts(&doc, &selection, &pairs, &contexts);
        let result = hook_with_context(&state, '(');

        assert!(result.is_some());
        let transaction = result.unwrap();
        let mut new_doc = doc.clone();
        assert!(transaction.apply(&mut new_doc));
        // Should only insert the typed char, NOT the pair
        assert_eq!(new_doc.to_string(), "test(\n");
    }

    #[test]
    fn test_hook_with_context_allowed_in_string() {
        // Use a space before cursor so quotes will pair (not after alphanumeric)
        let doc = Rope::from("test \n");
        let selection = Selection::single(5, 6);
        // Quote allowed in CODE and STRING contexts
        let pairs = BracketSet::new(vec![BracketPair::new("'", "'")
            .with_kind(BracketKind::Quote)
            .with_contexts(ContextMask::CODE | ContextMask::STRING)]);
        // We're in a STRING context
        let contexts = vec![BracketContext::String];

        let state = AutoPairState::with_contexts(&doc, &selection, &pairs, &contexts);
        let result = hook_with_context(&state, '\'');

        assert!(result.is_some());
        let transaction = result.unwrap();
        let mut new_doc = doc.clone();
        assert!(transaction.apply(&mut new_doc));
        // Should insert the pair since quotes are allowed in strings
        assert_eq!(new_doc.to_string(), "test ''\n");
    }

    #[test]
    fn test_hook_with_context_no_context_defaults_to_code() {
        let doc = Rope::from("test\n");
        let selection = Selection::single(4, 5);
        let pairs = BracketSet::new(vec![
            BracketPair::new("(", ")").with_contexts(ContextMask::CODE)
        ]);

        // No contexts provided - should default to CODE
        let state = AutoPairState::new(&doc, &selection, &pairs);
        let result = hook_with_context(&state, '(');

        assert!(result.is_some());
        let transaction = result.unwrap();
        let mut new_doc = doc.clone();
        assert!(transaction.apply(&mut new_doc));
        assert_eq!(new_doc.to_string(), "test()\n");
    }

    #[test]
    fn test_hook_with_context_multi_cursor_different_contexts() {
        // Original: "code \"str\" code\n"
        // Positions: c(0) o(1) d(2) e(3) " "(4) \"(5) s(6) t(7) r(8) \"(9) ...
        let doc = Rope::from("code \"str\" code\n");
        // Two cursors: one in code (pos 4 = space), one in string (pos 7 = 't')
        let selection = Selection::new(smallvec::smallvec![Range::new(4, 5), Range::new(7, 8)], 0);
        let pairs = BracketSet::new(vec![
            BracketPair::new("(", ")").with_contexts(ContextMask::CODE)
        ]);
        // First cursor in CODE, second in STRING
        let contexts = vec![BracketContext::Code, BracketContext::String];

        let state = AutoPairState::with_contexts(&doc, &selection, &pairs, &contexts);
        let result = hook_with_context(&state, '(');

        assert!(result.is_some());
        let transaction = result.unwrap();
        let mut new_doc = doc.clone();
        assert!(transaction.apply(&mut new_doc));
        // First cursor (pos 4) gets pair "()" in CODE context
        // Second cursor (pos 7) gets just "(" in STRING context (pair not allowed)
        // After first insert at pos 4: "code() \"str\" code\n" (inserted 2 chars)
        // After second insert at pos 7+2=9: "code() \"s(tr\" code\n" (inserted 1 char)
        assert_eq!(new_doc.to_string(), "code() \"s(tr\" code\n");
    }

    #[test]
    fn test_surround_pairs_iterator() {
        let pairs = vec![
            BracketPair::new("(", ")").with_surround(true),
            BracketPair::new("{%", "%}").with_surround(false),
            BracketPair::new("[", "]").with_surround(true),
        ];
        let set = BracketSet::new(pairs);

        let surround: Vec<_> = set.surround_pairs().collect();
        assert_eq!(surround.len(), 2);
        assert_eq!(surround[0].open, "(");
        assert_eq!(surround[1].open, "[");
    }

    #[test]
    fn test_get_pair_by_char_open() {
        let set = BracketSet::from_default_pairs();

        // Look up by open char
        let result = set.get_pair_by_char('(');
        assert!(result.is_some());
        let (open, close) = result.unwrap();
        assert_eq!(open, "(");
        assert_eq!(close, ")");
    }

    #[test]
    fn test_get_pair_by_char_close() {
        let set = BracketSet::from_default_pairs();

        // Look up by close char
        let result = set.get_pair_by_char(')');
        assert!(result.is_some());
        let (open, close) = result.unwrap();
        assert_eq!(open, "(");
        assert_eq!(close, ")");
    }

    #[test]
    fn test_get_pair_by_char_symmetric() {
        let set = BracketSet::from_default_pairs();

        // Symmetric pairs (quotes)
        let result = set.get_pair_by_char('"');
        assert!(result.is_some());
        let (open, close) = result.unwrap();
        assert_eq!(open, "\"");
        assert_eq!(close, "\"");
    }

    #[test]
    fn test_get_pair_by_char_not_found() {
        let set = BracketSet::from_default_pairs();

        // Character not in any pair
        let result = set.get_pair_by_char('x');
        assert!(result.is_none());
    }

    #[test]
    fn test_get_pair_by_char_only_surround_pairs() {
        let pairs = vec![
            BracketPair::new("(", ")").with_surround(true),
            BracketPair::new("{%", "%}").with_surround(false),
        ];
        let set = BracketSet::new(pairs);

        // ( is a surround pair
        let result = set.get_pair_by_char('(');
        assert!(result.is_some());

        // { starts the non-surround pair, should not be found
        let result = set.get_pair_by_char('{');
        assert!(result.is_none());
    }

    #[test]
    fn test_get_pair_by_char_multi_char() {
        let pairs = vec![
            BracketPair::new("{", "}").with_surround(true),
            BracketPair::new("{%", "%}").with_surround(true),
        ];
        let set = BracketSet::new(pairs);

        // { should match single-char pair
        let result = set.get_pair_by_char('{');
        assert!(result.is_some());
        let (open, close) = result.unwrap();
        assert_eq!(open, "{");
        assert_eq!(close, "}");
    }

    #[test]
    fn test_get_surround_strings_fallback() {
        let set = BracketSet::from_default_pairs();

        // Known pair
        let (open, close) = set.get_surround_strings('(');
        assert_eq!(open, "(");
        assert_eq!(close, ")");

        // Unknown char - falls back to same char
        let (open, close) = set.get_surround_strings('x');
        assert_eq!(open, "x");
        assert_eq!(close, "x");
    }
}
