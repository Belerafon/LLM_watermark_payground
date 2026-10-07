  if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !els.generateBtn.disabled) {
    els.generateBtn.click();
  }
});

/* ── init ── */
const hintPop = document.getElementById("hint-pop");
let hintAnchor = null;
let hintHideTimer;
function tipHost(node) {
  const el = node?.closest?.("[data-tip], label");
  if (!el) return null;
  if (el.dataset.tip) return el;
  const inner = el.querySelector("[data-tip]");
  return inner?.dataset.tip ? inner : null;
}
function placeHint(anchor) {
  const text = anchor.dataset.tip;
  if (!text) return;
  clearTimeout(hintHideTimer);
  hintAnchor = anchor;
  hintPop.hidden = false;
  hintPop.textContent = text;
  const margin = 8;
  const r = anchor.getBoundingClientRect();
  const w = hintPop.offsetWidth;
  const h = hintPop.offsetHeight;
  let left = Math.min(r.left, window.innerWidth - margin - w);
  left = Math.max(margin, left);
  let top = r.top - h - margin;
  if (top < margin) top = Math.min(r.bottom + margin, window.innerHeight - margin - h);
  top = Math.max(margin, top);
  hintPop.style.left = `${left}px`;
  hintPop.style.top = `${top}px`;
}
function hideHint() {
  clearTimeout(hintHideTimer);
  hintPop.hidden = true;
  hintAnchor = null;
}
document.addEventListener("mouseover", (e) => {
  if (hintPop.contains(e.target)) {
    clearTimeout(hintHideTimer);
    return;
  }
  const host = tipHost(e.target);
  if (!host) return;
  placeHint(host);
});
document.addEventListener("mouseout", (e) => {
  const host = tipHost(e.target);
  if (!host && !hintPop.contains(e.target)) return;
  if (hintAnchor?.contains(e.relatedTarget) || hintPop.contains(e.relatedTarget)) return;
  // Allow crossing the gap into a long tooltip to scroll its text.
  hintHideTimer = setTimeout(hideHint, 180);
});
document.addEventListener("focusin", (e) => {
  const host = e.target.closest?.("[data-tip]");
  if (host?.dataset.tip) placeHint(host);
});
document.addEventListener("focusout", hideHint);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") hideHint();
});
window.addEventListener("scroll", (e) => {
  if (e.target === hintPop) return;
  if (hintAnchor?.contains(document.activeElement)) placeHint(hintAnchor);
  else hideHint();
}, true);
window.addEventListener("resize", hideHint);

syncParamVisibility();
refreshModelOptions(lastBackend);
updateButtons();
setStatus("Выберите модель: нажмите «Скачать и загрузить» или загрузите уже скачанную из кэша.");
refreshCachedModels();
