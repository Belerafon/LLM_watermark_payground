import test from "node:test";
import assert from "node:assert/strict";
import { displayParts } from "../output-text.js";

test("a detector chip cap never truncates the edited Arkanoid answer", () => {
  const edited = "def draw_objects():\n" + "    draw_brick()\n".repeat(140)
    + '    text = "ПОБЕДА! Нажмите R для перезапуска"\n'
    + "pygame.quit()\nsys.exit()";
  const chips = [...edited.slice(0, 2000)].map((text) => ({ text }));
  const parts = displayParts(chips, edited);
  assert.equal(parts.colouredTokens.map((token) => token.text).join("") + parts.remainder, edited);
  assert.match(parts.remainder, /sys\.exit\(\)$/);
});

test("tokenizer normalisation does not rewrite the edited text", () => {
  const edited = "    text = \"ПОБЕДА!\"\n";
  const parts = displayParts([{ text: " text" }], edited);
  assert.deepEqual(parts, { colouredTokens: [], remainder: edited });
});
