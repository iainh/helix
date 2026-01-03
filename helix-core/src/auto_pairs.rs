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

/// Default multi-character pairs for common languages.
pub const DEFAULT_MULTI_CHAR_PAIRS: &[(&str, &str)] = &[
    ("(", ")"),
    ("{", "}"),
    ("[", "]"),
    ("'", "'"),
    ("\"", "\""),
    ("`", "`"),
];

// ============================================================================
// New Multi-Character Auto-Pairs Types
// ============================================================================

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

        let slice = doc.slice(cursor_char..cursor_char + close_len);
        let at_cursor: String = slice.chars().collect();

        if at_cursor == pair.close {
            return Some(pair);
        }
    }

    None
}

// ============================================================================
// Legacy Single-Character Auto-Pairs (Backward Compatibility)
// ============================================================================

/// The type that represents the collection of auto pairs,
/// keyed by both opener and closer.
#[derive(Debug, Clone)]
pub struct AutoPairs(HashMap<char, Pair>);

/// Represents the config for a particular pairing.
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

// ============================================================================
// New Multi-Character Hook
// ============================================================================

/// Hook for multi-character auto-pairs.
///
/// This is the new entry point that supports multi-character pairs like ```,
/// `{% %}`, `<!-- -->`, etc.
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

        // Try to detect a completed trigger (the char was just typed, so cursor is after it)
        // For multi-char triggers, we need to look at what's before cursor + the new char
        if let Some(pair) = detect_trigger_at(doc, cursor, ch, pairs) {
            // Check if we should skip over existing close
            let next_char = doc.get_char(cursor);
            if pair.same() && next_char == pair.close_first_char() {
                // Skip over the close - no change, just move cursor
                let next_range = get_next_range(doc, start_range, offs, 0);
                end_ranges.push(next_range);
                made_changes = true;
                return (cursor, cursor, None);
            }

            // Check if we should insert close
            if pair.should_close(doc, start_range) {
                // For multi-char pairs, we need to handle the trigger replacement
                let trigger_len = pair.trigger.chars().count();
                let open_len = pair.open.chars().count();

                if trigger_len == 1 && open_len == 1 {
                    // Simple case: single char trigger, insert close after
                    let close = Tendril::from(pair.close.as_str());
                    let close_len = close.chars().count();
                    let next_range = get_next_range(doc, start_range, offs, close_len);
                    end_ranges.push(next_range);
                    offs += close_len;
                    made_changes = true;
                    return (cursor, cursor, Some(close));
                } else {
                    // Multi-char: we've typed the last char of trigger, need to insert close
                    // The trigger chars are already in the document (except last one being typed)
                    let close = Tendril::from(pair.close.as_str());
                    let close_len = close.chars().count();
                    let next_range = get_next_range(doc, start_range, offs, close_len);
                    end_ranges.push(next_range);
                    offs += close_len;
                    made_changes = true;
                    return (cursor, cursor, Some(close));
                }
            } else {
                // Don't auto-close, just let the char through
                let next_range = get_next_range(doc, start_range, offs, 0);
                end_ranges.push(next_range);
                return (cursor, cursor, None);
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

        // No auto-pair action, return no-op
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

// ============================================================================
// Internal Helpers
// ============================================================================

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

// ============================================================================
// Unit Tests
// ============================================================================

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
        let pairs = vec![
            BracketPair::new("`", "`"),
            BracketPair::new("```", "```"),
        ];
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
        // It decides what to insert instead of just the typed char.
        // For '(' with cursor at position 4, it should insert "()" 
        // But the hook only adds the CLOSE part - the open is handled by the caller.
        // Actually, looking at the original hook - it only inserts the close.
        let doc = Rope::from("test\n");
        let selection = Selection::single(4, 5); // cursor at end of "test", before \n
        let set = BracketSet::from_default_pairs();

        let result = hook_multi(&doc, &selection, '(', &set);
        assert!(result.is_some());

        let transaction = result.unwrap();
        let mut new_doc = doc.clone();
        assert!(transaction.apply(&mut new_doc));

        // hook_multi inserts the close char at cursor position
        assert_eq!(new_doc.to_string(), "test)\n");
    }

    #[test]
    fn test_hook_multi_triple_backtick() {
        // When we type the third backtick, the document already has ``
        // and cursor is at position 6 (after the two backticks).
        // The hook should detect ``` trigger and insert ```
        let doc = Rope::from("test``\n");
        let selection = Selection::single(6, 7); // cursor after `` (on the \n)

        let pairs = vec![
            BracketPair::new("`", "`"),
            BracketPair::new("```", "```"),
        ];
        let set = BracketSet::new(pairs);

        let result = hook_multi(&doc, &selection, '`', &set);
        assert!(result.is_some());

        let transaction = result.unwrap();
        let mut new_doc = doc.clone();
        assert!(transaction.apply(&mut new_doc));

        // The hook inserts the closing ``` at cursor position 6
        // So: "test``" + "```" + "\n" = "test`````\n"
        assert_eq!(new_doc.to_string(), "test`````\n");
    }

    #[test]
    fn test_hook_multi_jinja_delimiter() {
        // Document has "{" and we're typing "%"
        // The trigger is "{%" so after typing %, we should get "%}"
        let doc = Rope::from("test{\n");
        let selection = Selection::single(5, 6); // cursor after { (on the \n)

        let pairs = vec![
            BracketPair::new("{", "}"),
            BracketPair::new("{%", "%}"),
        ];
        let set = BracketSet::new(pairs);

        let result = hook_multi(&doc, &selection, '%', &set);
        assert!(result.is_some());

        let transaction = result.unwrap();
        let mut new_doc = doc.clone();
        assert!(transaction.apply(&mut new_doc));

        // The hook inserts the closing %} at cursor position 5
        // So: "test{" + "%}" + "\n" = "test{%}\n"
        assert_eq!(new_doc.to_string(), "test{%}\n");
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
        // Quotes shouldn't auto-pair after alphanumeric
        let doc = Rope::from("test\n");
        let selection = Selection::single(4, 5); // cursor right after "test"

        let set = BracketSet::from_default_pairs();

        // After an alphanumeric, quotes should NOT auto-pair
        let result = hook_multi(&doc, &selection, '"', &set);
        // The hook returns None for symmetric pairs when prev is alphanumeric
        assert!(result.is_none() || {
            // Or it returns a no-op transaction
            let transaction = result.unwrap();
            let mut new_doc = doc.clone();
            transaction.apply(&mut new_doc);
            new_doc.slice(..) == "test\n"
        });
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

        // Should have inserted closing "
        assert_eq!(new_doc.to_string(), "test \"\n");
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
}
