use std::fmt::Display;

use crate::{
    graphemes::next_grapheme_boundary,
    match_brackets::{
        find_matching_bracket, find_matching_bracket_fuzzy, get_pair, is_close_bracket,
        is_open_bracket,
    },
    movement::Direction,
    search, Range, Selection, Syntax,
};
use ropey::RopeSlice;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    PairNotFound,
    CursorOverlap,
    RangeExceedsText,
    CursorOnAmbiguousPair,
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match *self {
            Error::PairNotFound => "Surround pair not found around all cursors",
            Error::CursorOverlap => "Cursors overlap for a single surround pair range",
            Error::RangeExceedsText => "Cursor range exceeds text length",
            Error::CursorOnAmbiguousPair => "Cursor on ambiguous surround pair",
        })
    }
}

type Result<T> = std::result::Result<T, Error>;

/// Finds the position of surround pairs of any [`crate::match_brackets::PAIRS`]
/// using tree-sitter when possible.
///
/// # Returns
///
/// Tuple `(anchor, head)`, meaning it is not always ordered.
pub fn find_nth_closest_pairs_pos(
    syntax: Option<&Syntax>,
    text: RopeSlice,
    range: Range,
    skip: usize,
) -> Result<(usize, usize)> {
    match syntax {
        Some(syntax) => find_nth_closest_pairs_ts(syntax, text, range, skip),
        None => find_nth_closest_pairs_plain(text, range, skip),
    }
}

fn find_nth_closest_pairs_ts(
    syntax: &Syntax,
    text: RopeSlice,
    range: Range,
    mut skip: usize,
) -> Result<(usize, usize)> {
    let mut opening = range.from();
    // We want to expand the selection if we are already on the found pair,
    // otherwise we would need to subtract "-1" from "range.to()".
    let mut closing = range.to();

    while skip > 0 {
        closing = find_matching_bracket_fuzzy(syntax, text, closing).ok_or(Error::PairNotFound)?;
        opening = find_matching_bracket(syntax, text, closing).ok_or(Error::PairNotFound)?;
        // If we're already on a closing bracket "find_matching_bracket_fuzzy" will return
        // the position of the opening bracket.
        if closing < opening {
            (opening, closing) = (closing, opening);
        }

        // In case found brackets are partially inside current selection.
        if range.from() < opening || closing < range.to() - 1 {
            closing = next_grapheme_boundary(text, closing);
        } else {
            skip -= 1;
            if skip != 0 {
                closing = next_grapheme_boundary(text, closing);
            }
        }
    }

    // Keep the original direction.
    if let Direction::Forward = range.direction() {
        Ok((opening, closing))
    } else {
        Ok((closing, opening))
    }
}

fn find_nth_closest_pairs_plain(
    text: RopeSlice,
    range: Range,
    mut skip: usize,
) -> Result<(usize, usize)> {
    let mut stack = Vec::with_capacity(2);
    let pos = range.from();
    let mut close_pos = pos.saturating_sub(1);

    for ch in text.chars_at(pos) {
        close_pos += 1;

        if is_open_bracket(ch) {
            // Track open pairs encountered so that we can step over
            // the corresponding close pairs that will come up further
            // down the loop. We want to find a lone close pair whose
            // open pair is before the cursor position.
            stack.push(ch);
            continue;
        }

        if !is_close_bracket(ch) {
            // We don't care if this character isn't a brace pair item,
            // so short circuit here.
            continue;
        }

        let (open, close) = get_pair(ch);

        if stack.last() == Some(&open) {
            // If we are encountering the closing pair for an opener
            // we just found while traversing, then its inside the
            // selection and should be skipped over.
            stack.pop();
            continue;
        }

        match find_nth_open_pair(text, open, close, close_pos, 1) {
            // Before we accept this pair, we want to ensure that the
            // pair encloses the range rather than just the cursor.
            Some(open_pos)
                if open_pos <= pos.saturating_add(1)
                    && close_pos >= range.to().saturating_sub(1) =>
            {
                // Since we have special conditions for when to
                // accept, we can't just pass the skip parameter on
                // through to the find_nth_*_pair methods, so we
                // track skips manually here.
                if skip > 1 {
                    skip -= 1;
                    continue;
                }

                return match range.direction() {
                    Direction::Forward => Ok((open_pos, close_pos)),
                    Direction::Backward => Ok((close_pos, open_pos)),
                };
            }
            _ => continue,
        }
    }

    Err(Error::PairNotFound)
}

/// Find the position of surround pairs of `ch` which can be either a closing
/// or opening pair. `n` will skip n - 1 pairs (eg. n=2 will discard (only)
/// the first pair found and keep looking)
pub fn find_nth_pairs_pos(
    text: RopeSlice,
    ch: char,
    range: Range,
    n: usize,
) -> Result<(usize, usize)> {
    if text.len_chars() < 2 {
        return Err(Error::PairNotFound);
    }
    if range.to() >= text.len_chars() {
        return Err(Error::RangeExceedsText);
    }

    let (open, close) = get_pair(ch);
    let pos = range.cursor(text);

    let (open, close) = if open == close {
        if Some(open) == text.get_char(pos) {
            // Cursor is directly on match char. We return no match
            // because there's no way to know which side of the char
            // we should be searching on.
            return Err(Error::CursorOnAmbiguousPair);
        }
        (
            search::find_nth_prev(text, open, pos, n),
            search::find_nth_next(text, close, pos, n),
        )
    } else {
        (
            find_nth_open_pair(text, open, close, pos, n),
            find_nth_close_pair(text, open, close, pos, n),
        )
    };

    // preserve original direction
    match range.direction() {
        Direction::Forward => Option::zip(open, close).ok_or(Error::PairNotFound),
        Direction::Backward => Option::zip(close, open).ok_or(Error::PairNotFound),
    }
}

fn find_nth_open_pair(
    text: RopeSlice,
    open: char,
    close: char,
    mut pos: usize,
    n: usize,
) -> Option<usize> {
    if pos >= text.len_chars() {
        return None;
    }

    let mut chars = text.chars_at(pos + 1);

    // Adjusts pos for the first iteration, and handles the case of the
    // cursor being *on* the close character which will get falsely stepped over
    // if not skipped here
    if chars.prev()? == open {
        return Some(pos);
    }

    for _ in 0..n {
        let mut step_over: usize = 0;

        loop {
            let c = chars.prev()?;
            pos = pos.saturating_sub(1);

            // ignore other surround pairs that are enclosed *within* our search scope
            if c == close {
                step_over += 1;
            } else if c == open {
                if step_over == 0 {
                    break;
                }

                step_over = step_over.saturating_sub(1);
            }
        }
    }

    Some(pos)
}

fn find_nth_close_pair(
    text: RopeSlice,
    open: char,
    close: char,
    mut pos: usize,
    n: usize,
) -> Option<usize> {
    if pos >= text.len_chars() {
        return None;
    }

    let mut chars = text.chars_at(pos);

    if chars.next()? == close {
        return Some(pos);
    }

    for _ in 0..n {
        let mut step_over: usize = 0;

        loop {
            let c = chars.next()?;
            pos += 1;

            if c == open {
                step_over += 1;
            } else if c == close {
                if step_over == 0 {
                    break;
                }

                step_over = step_over.saturating_sub(1);
            }
        }
    }

    Some(pos)
}

/// Find the position of surround pairs using multi-character open/close strings.
///
/// Similar to `find_nth_pairs_pos` but works with string patterns instead of single chars.
/// Returns the starting positions of the open and close strings.
pub fn find_nth_pairs_pos_multi(
    text: RopeSlice,
    open: &str,
    close: &str,
    range: Range,
    n: usize,
) -> Result<(usize, usize)> {
    if text.len_chars() < open.len() + close.len() {
        return Err(Error::PairNotFound);
    }
    if range.to() >= text.len_chars() {
        return Err(Error::RangeExceedsText);
    }

    let pos = range.cursor(text);

    let (open_pos, close_pos) = if open == close {
        // Symmetric pair (like ```)
        (
            find_nth_prev_str(text, open, pos, n),
            find_nth_next_str(text, close, pos, n),
        )
    } else {
        // Asymmetric pair (like {% %})
        (
            find_nth_open_pair_multi(text, open, close, pos, n),
            find_nth_close_pair_multi(text, open, close, pos, n),
        )
    };

    match range.direction() {
        Direction::Forward => Option::zip(open_pos, close_pos).ok_or(Error::PairNotFound),
        Direction::Backward => Option::zip(close_pos, open_pos).ok_or(Error::PairNotFound),
    }
}

/// Find nth occurrence of a string pattern searching backward from pos.
fn find_nth_prev_str(text: RopeSlice, pattern: &str, pos: usize, n: usize) -> Option<usize> {
    let pattern_len = pattern.chars().count();
    if pattern_len == 0 || pos < pattern_len {
        return None;
    }

    let mut found = 0;
    let mut search_pos = pos.saturating_sub(pattern_len);

    loop {
        // Check if pattern matches at search_pos
        if search_pos + pattern_len <= text.len_chars()
            && text.slice(search_pos..search_pos + pattern_len) == pattern
        {
            found += 1;
            if found == n {
                return Some(search_pos);
            }
        }

        if search_pos == 0 {
            break;
        }
        search_pos -= 1;
    }

    None
}

/// Find nth occurrence of a string pattern searching forward from pos.
fn find_nth_next_str(text: RopeSlice, pattern: &str, pos: usize, n: usize) -> Option<usize> {
    let pattern_len = pattern.chars().count();
    if pattern_len == 0 {
        return None;
    }

    let mut found = 0;
    let mut search_pos = pos;

    while search_pos + pattern_len <= text.len_chars() {
        if text.slice(search_pos..search_pos + pattern_len) == pattern {
            found += 1;
            if found == n {
                return Some(search_pos);
            }
        }
        search_pos += 1;
    }

    None
}

/// Find the nth opening pair position, handling nesting.
fn find_nth_open_pair_multi(
    text: RopeSlice,
    open: &str,
    close: &str,
    pos: usize,
    n: usize,
) -> Option<usize> {
    let open_len = open.chars().count();
    let close_len = close.chars().count();

    let mut found = 0;
    let mut step_over: usize = 0;

    // Check if we're starting on or within the open pattern
    // For multi-char patterns, the cursor might be anywhere within the pattern
    for offset in 0..open_len {
        let start = pos.saturating_sub(offset);
        if start + open_len <= text.len_chars() {
            if text.slice(start..start + open_len) == open {
                return Some(start);
            }
        }
    }

    // Check if we're on the close pattern - if so, start searching before it
    let mut search_pos = pos;
    for offset in 0..close_len {
        let start = pos.saturating_sub(offset);
        if start + close_len <= text.len_chars() {
            if text.slice(start..start + close_len) == close {
                // We're on the close pattern, start search before it
                search_pos = start;
                break;
            }
        }
    }

    while search_pos > 0 {
        search_pos -= 1;

        // Check for close pattern (increases nesting)
        if search_pos + close_len <= text.len_chars() {
            if text.slice(search_pos..search_pos + close_len) == close {
                step_over += 1;
                continue;
            }
        }

        // Check for open pattern
        if search_pos + open_len <= text.len_chars() {
            if text.slice(search_pos..search_pos + open_len) == open {
                if step_over == 0 {
                    found += 1;
                    if found == n {
                        return Some(search_pos);
                    }
                } else {
                    step_over -= 1;
                }
            }
        }
    }

    None
}

/// Find the nth closing pair position, handling nesting.
fn find_nth_close_pair_multi(
    text: RopeSlice,
    open: &str,
    close: &str,
    pos: usize,
    n: usize,
) -> Option<usize> {
    let open_len = open.chars().count();
    let close_len = close.chars().count();

    let mut found = 0;
    let mut step_over: usize = 0;
    let mut search_pos = pos;

    // Check if we're starting on or within the close pattern
    // For multi-char patterns, the cursor might be anywhere within the pattern
    for offset in 0..close_len {
        let start = pos.saturating_sub(offset);
        if start + close_len <= text.len_chars() {
            if text.slice(start..start + close_len) == close {
                return Some(start);
            }
        }
    }

    while search_pos + 1 < text.len_chars() {
        search_pos += 1;

        // Check for open pattern (increases nesting)
        if search_pos + open_len <= text.len_chars() {
            if text.slice(search_pos..search_pos + open_len) == open {
                step_over += 1;
                continue;
            }
        }

        // Check for close pattern
        if search_pos + close_len <= text.len_chars() {
            if text.slice(search_pos..search_pos + close_len) == close {
                if step_over == 0 {
                    found += 1;
                    if found == n {
                        return Some(search_pos);
                    }
                } else {
                    step_over -= 1;
                }
            }
        }
    }

    None
}

/// Find position of surround characters around every cursor for multi-char pairs.
///
/// Returns a Vec of (position, length) tuples for deletion/replacement.
/// The Vec contains pairs: [(open_pos, open_len), (close_pos, close_len), ...]
pub fn get_surround_pos_multi(
    text: RopeSlice,
    selection: &Selection,
    open: &str,
    close: &str,
    skip: usize,
) -> Result<Vec<(usize, usize)>> {
    let open_len = open.chars().count();
    let close_len = close.chars().count();
    let mut change_pos = Vec::new();

    for &range in selection {
        let (open_pos, close_pos) = find_nth_pairs_pos_multi(text, open, close, range, skip)?;
        let (open_pos, close_pos) = (open_pos.min(close_pos), open_pos.max(close_pos));

        // Check for overlaps
        for &(pos, _) in &change_pos {
            if pos == open_pos || pos == close_pos {
                return Err(Error::CursorOverlap);
            }
        }

        change_pos.push((open_pos, open_len));
        change_pos.push((close_pos, close_len));
    }

    Ok(change_pos)
}

/// Find position of surround characters around every cursor. Returns None
/// if any positions overlap. Note that the positions are in a flat Vec.
/// Use get_surround_pos().chunks(2) to get matching pairs of surround positions.
/// `ch` can be either closing or opening pair. If `ch` is None, surround pairs
/// are automatically detected around each cursor (note that this may result
/// in them selecting different surround characters for each selection).
pub fn get_surround_pos(
    syntax: Option<&Syntax>,
    text: RopeSlice,
    selection: &Selection,
    ch: Option<char>,
    skip: usize,
) -> Result<Vec<usize>> {
    let mut change_pos = Vec::new();

    for &range in selection {
        let (open_pos, close_pos) = {
            let range_raw = match ch {
                Some(ch) => find_nth_pairs_pos(text, ch, range, skip)?,
                None => find_nth_closest_pairs_pos(syntax, text, range, skip)?,
            };
            let range = Range::new(range_raw.0, range_raw.1);
            (range.from(), range.to())
        };
        if change_pos.contains(&open_pos) || change_pos.contains(&close_pos) {
            return Err(Error::CursorOverlap);
        }
        // ensure the positions are always paired in the forward direction
        change_pos.extend_from_slice(&[open_pos.min(close_pos), close_pos.max(open_pos)]);
    }
    Ok(change_pos)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Range;

    use ropey::Rope;
    use smallvec::SmallVec;

    #[test]
    fn test_get_surround_pos() {
        #[rustfmt::skip]
        let (doc, selection, expectations) =
            rope_with_selections_and_expectations(
                "(some) (chars)\n(newline)",
                "_ ^  _ _ ^   _\n_    ^  _"
            );

        assert_eq!(
            get_surround_pos(None, doc.slice(..), &selection, Some('('), 1).unwrap(),
            expectations
        );
    }

    #[test]
    fn test_get_surround_pos_bail_different_surround_chars() {
        #[rustfmt::skip]
        let (doc, selection, _) =
            rope_with_selections_and_expectations(
                "[some]\n(chars)xx\n(newline)",
                "  ^   \n  ^      \n         "
            );

        assert_eq!(
            get_surround_pos(None, doc.slice(..), &selection, Some('('), 1),
            Err(Error::PairNotFound)
        );
    }

    #[test]
    fn test_get_surround_pos_bail_overlapping_surround_chars() {
        #[rustfmt::skip]
        let (doc, selection, _) =
            rope_with_selections_and_expectations(
                "[some]\n(chars)xx\n(newline)",
                "      \n       ^ \n      ^  "
            );

        assert_eq!(
            get_surround_pos(None, doc.slice(..), &selection, Some('('), 1),
            Err(Error::PairNotFound) // overlapping surround chars
        );
    }

    #[test]
    fn test_get_surround_pos_bail_cursor_overlap() {
        #[rustfmt::skip]
        let (doc, selection, _) =
            rope_with_selections_and_expectations(
                "[some]\n(chars)xx\n(newline)",
                "  ^^  \n         \n         "
            );

        assert_eq!(
            get_surround_pos(None, doc.slice(..), &selection, Some('['), 1),
            Err(Error::CursorOverlap)
        );
    }

    #[test]
    fn test_find_nth_pairs_pos_quote_success() {
        #[rustfmt::skip]
        let (doc, selection, expectations) =
            rope_with_selections_and_expectations(
                "some 'quoted text' on this 'line'\n'and this one'",
                "     _        ^  _               \n              "
            );

        assert_eq!(2, expectations.len());
        assert_eq!(
            find_nth_pairs_pos(doc.slice(..), '\'', selection.primary(), 1)
                .expect("find should succeed"),
            (expectations[0], expectations[1])
        )
    }

    #[test]
    fn test_find_nth_pairs_pos_nested_quote_success() {
        #[rustfmt::skip]
        let (doc, selection, expectations) =
            rope_with_selections_and_expectations(
                "some 'nested 'quoted' text' on this 'line'\n'and this one'",
                "     _           ^        _               \n              "
            );

        assert_eq!(2, expectations.len());
        assert_eq!(
            find_nth_pairs_pos(doc.slice(..), '\'', selection.primary(), 2)
                .expect("find should succeed"),
            (expectations[0], expectations[1])
        )
    }

    #[test]
    fn test_find_nth_pairs_pos_inside_quote_ambiguous() {
        #[rustfmt::skip]
        let (doc, selection, _) =
            rope_with_selections_and_expectations(
                "some 'nested 'quoted' text' on this 'line'\n'and this one'",
                "                    ^                     \n              "
            );

        assert_eq!(
            find_nth_pairs_pos(doc.slice(..), '\'', selection.primary(), 1),
            Err(Error::CursorOnAmbiguousPair)
        )
    }

    #[test]
    fn test_find_nth_closest_pairs_pos_index_range_panic() {
        #[rustfmt::skip]
        let (doc, selection, _) =
            rope_with_selections_and_expectations(
                "(a)c)",
                "^^^^^"
            );

        assert_eq!(
            find_nth_closest_pairs_pos(None, doc.slice(..), selection.primary(), 1),
            Err(Error::PairNotFound)
        )
    }

    // Create a Rope and a matching Selection using a specification language.
    // ^ is a single-point selection.
    // _ is an expected index. These are returned as a Vec<usize> for use in assertions.
    fn rope_with_selections_and_expectations(
        text: &str,
        spec: &str,
    ) -> (Rope, Selection, Vec<usize>) {
        if text.len() != spec.len() {
            panic!("specification must match text length -- are newlines aligned?");
        }

        let rope = Rope::from(text);

        let selections: SmallVec<[Range; 1]> = spec
            .match_indices('^')
            .map(|(i, _)| Range::point(i))
            .collect();

        let expectations: Vec<usize> = spec.match_indices('_').map(|(i, _)| i).collect();

        (rope, Selection::new(selections, 0), expectations)
    }

    #[test]
    fn test_find_nth_pairs_pos_multi_simple() {
        let doc = Rope::from("{% hello %}");
        let range = Range::point(5); // cursor on 'e'

        let result = find_nth_pairs_pos_multi(doc.slice(..), "{%", "%}", range, 1);
        assert!(result.is_ok());
        let (open_pos, close_pos) = result.unwrap();
        assert_eq!(open_pos, 0); // {% starts at 0
        assert_eq!(close_pos, 9); // %} starts at 9
    }

    #[test]
    fn test_find_nth_pairs_pos_multi_nested() {
        // "{% outer {% inner %} outer %}"
        //  0         1         2
        //  0123456789012345678901234567890
        // Inner {% at 9, inner %} at 18
        // Outer {% at 0, outer %} at 27
        let doc = Rope::from("{% outer {% inner %} outer %}");
        let range = Range::point(14); // cursor on 'n' in inner

        // First pair should be inner
        let result = find_nth_pairs_pos_multi(doc.slice(..), "{%", "%}", range, 1);
        assert!(result.is_ok());
        let (open_pos, close_pos) = result.unwrap();
        assert_eq!(open_pos, 9); // inner {% starts at 9
        assert_eq!(close_pos, 18); // inner %} starts at 18

        // Second pair should be outer
        let result = find_nth_pairs_pos_multi(doc.slice(..), "{%", "%}", range, 2);
        assert!(result.is_ok());
        let (open_pos, close_pos) = result.unwrap();
        assert_eq!(open_pos, 0); // outer {% starts at 0
        assert_eq!(close_pos, 27); // outer %} starts at 27
    }

    #[test]
    fn test_find_nth_pairs_pos_multi_symmetric() {
        let doc = Rope::from("```code block```");
        let range = Range::point(6); // cursor on 'e'

        let result = find_nth_pairs_pos_multi(doc.slice(..), "```", "```", range, 1);
        assert!(result.is_ok());
        let (open_pos, close_pos) = result.unwrap();
        assert_eq!(open_pos, 0);
        assert_eq!(close_pos, 13);
    }

    #[test]
    fn test_find_nth_pairs_pos_multi_not_found() {
        let doc = Rope::from("no pairs here");
        let range = Range::point(5);

        let result = find_nth_pairs_pos_multi(doc.slice(..), "{%", "%}", range, 1);
        assert_eq!(result, Err(Error::PairNotFound));
    }

    #[test]
    fn test_find_nth_pairs_pos_multi_cursor_at_end() {
        // Cursor at end of content but within pairs
        // Text: "{% hello %}" = 11 chars
        // Position 9 is on "%", position 10 is on "}"
        // %} starts at position 9
        let doc = Rope::from("{% hello %}");
        let range = Range::point(10); // Cursor on "}"

        let result = find_nth_pairs_pos_multi(doc.slice(..), "{%", "%}", range, 1);
        assert!(result.is_ok());
        let (open_pos, close_pos) = result.unwrap();
        assert_eq!(open_pos, 0);
        assert_eq!(close_pos, 9);
    }

    #[test]
    fn test_find_nth_pairs_pos_multi_cursor_past_end() {
        // Cursor position equals text length (end of document)
        let doc = Rope::from("{% hello %}");
        let range = Range::point(11); // At end

        let result = find_nth_pairs_pos_multi(doc.slice(..), "{%", "%}", range, 1);
        // Cursor at len_chars() is at the very end - should return RangeExceedsText
        // This matches the behavior of the single-char version
        assert_eq!(result, Err(Error::RangeExceedsText));
    }

    #[test]
    fn test_find_nth_pairs_pos_multi_html_comment() {
        let doc = Rope::from("<!-- this is a comment -->");
        let range = Range::point(10); // cursor on 'i'

        let result = find_nth_pairs_pos_multi(doc.slice(..), "<!--", "-->", range, 1);
        assert!(result.is_ok());
        let (open_pos, close_pos) = result.unwrap();
        assert_eq!(open_pos, 0);
        assert_eq!(close_pos, 23);
    }

    #[test]
    fn test_get_surround_pos_multi_single_cursor() {
        let doc = Rope::from("{% hello %}");
        let selection = Selection::single(5, 6); // cursor on 'e'

        let result = get_surround_pos_multi(doc.slice(..), &selection, "{%", "%}", 1);
        assert!(result.is_ok());
        let positions = result.unwrap();
        // Should return [(open_pos, open_len), (close_pos, close_len)]
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0], (0, 2)); // {% at 0, len 2
        assert_eq!(positions[1], (9, 2)); // %} at 9, len 2
    }

    #[test]
    fn test_get_surround_pos_multi_multiple_cursors() {
        let doc = Rope::from("{% a %} {% b %}");
        let selection = Selection::new(
            smallvec::smallvec![Range::point(3), Range::point(11)],
            0,
        );

        let result = get_surround_pos_multi(doc.slice(..), &selection, "{%", "%}", 1);
        assert!(result.is_ok());
        let positions = result.unwrap();
        assert_eq!(positions.len(), 4);
        assert_eq!(positions[0], (0, 2)); // first {% at 0
        assert_eq!(positions[1], (5, 2)); // first %} at 5
        assert_eq!(positions[2], (8, 2)); // second {% at 8
        assert_eq!(positions[3], (13, 2)); // second %} at 13
    }
}
