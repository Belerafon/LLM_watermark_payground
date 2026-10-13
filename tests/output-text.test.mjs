import test from "node:test";
import assert from "node:assert/strict";
import { displayParts } from "../output-text.js";

test("a long edited Arkanoid answer is coloured through its final token", () => {
  const edited = "def draw_objects():\n" + "    draw_brick()\n".repeat(140)
    + '    text = "ПОБЕДА! Нажмите R для перезапуска"\n'
    + "pygame.quit()\nsys.exit()";
  const chips = [...edited].map((text) => ({ text }));
  const parts = displayParts(chips, edited);
  assert.ok(parts.colouredTokens.length > 2000);
  assert.equal(parts.colouredTokens.map((token) => token.text).join(""), edited);
  assert.equal(parts.remainder, "");
});

test("tokenizer normalisation does not rewrite the edited text", () => {
  const edited = "    text = \"ПОБЕДА!\"\n";
  const parts = displayParts([{ text: " text" }], edited);
  assert.deepEqual(parts, { colouredTokens: [], remainder: edited });
});
