/** Keep the exact edited text when decoded token text differs from the input. */
export function displayParts(tokens, originalText) {
  const prefix = tokens.map((token) => token.text).join("");
  if (!originalText.startsWith(prefix)) {
    // A tokenizer can normalise whitespace; never replace the author's text.
    return { colouredTokens: [], remainder: originalText };
  }
  return { colouredTokens: tokens, remainder: originalText.slice(prefix.length) };
}
