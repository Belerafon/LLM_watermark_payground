const parts = ["main.part0.js", "main.part1.js", "main.part2.js", "main.part3.js", "main.part4.js", "main.part5.js"];
let code = "";
for (const p of parts) {
  const url = new URL(p, import.meta.url);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${p}: ${res.status}`);
  code += await res.text();
}
await import(URL.createObjectURL(new Blob([code], { type: "text/javascript" })));
