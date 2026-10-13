/** Keep the complete edited text when only a prefix of detector tokens is coloured. */
export function displayParts(tokens, originalText) {
  const prefix = tokens.map((token) => token.text).join("");
  if (!originalText.startsWith(prefix)) {
    // A tokenizer can normalise whitespace; never replace the author's text.
    return { colouredTokens: [], remainder: originalText };
  }
  return { colouredTokens: tokens, remainder: originalText.slice(prefix.length) };
}
