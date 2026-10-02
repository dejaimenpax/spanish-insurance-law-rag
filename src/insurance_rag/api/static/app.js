"use strict";

const $ = (id) => document.getElementById(id);
const form = $("ask-form");
const question = $("question");
const submit = $("submit");
const statusEl = $("status");
const answerEl = $("answer");
const answerText = $("answer-text");
const warningsEl = $("warnings");
const metaEl = $("meta");
const citedEl = $("cited");
const uncitedEl = $("uncited");
const othersEl = $("others");
const sourcesEmpty = $("sources-empty");

const DATE = new Intl.DateTimeFormat("es-ES", { day: "2-digit", month: "2-digit", year: "numeric" });
const formatDate = (iso) => (iso ? DATE.format(new Date(`${iso}T00:00:00`)) : "—");

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child !== null && child !== undefined) node.append(child);
  }
  return node;
}

async function loadNorms() {
  const container = $("norms");
  try {
    const response = await fetch("/norms");
    if (!response.ok) throw new Error(response.statusText);
    const norms = await response.json();
    container.replaceChildren(
      ...norms.map((norm) =>
        el(
          "label",
          { class: "chip", title: `${norm.title} · texto consolidado a ${formatDate(norm.consolidated_as_of)}` },
          el("input", { type: "checkbox", name: "norm", value: norm.id, id: `norm-${norm.id}` }),
          el("span", {}, norm.short_name),
        ),
      ),
    );
  } catch {
    container.replaceChildren(el("span", { class: "hint error" }, "No se han podido cargar las normas indexadas."));
  }
}

// --- Answer rendering ------------------------------------------------------------------------
// The answer is a list of segments: text, and citation markers inserted where Claude cited.
let segments = [];
let sources = [];

function renderAnswer(abstained = false) {
  answerText.classList.toggle("abstained", abstained);
  const paragraphs = [el("p")];
  for (const segment of segments) {
    if (segment.type === "text") {
      const parts = segment.text.split(/\n{2,}/);
      parts.forEach((part, i) => {
        if (i > 0) paragraphs.push(el("p"));
        paragraphs.at(-1).append(part);
      });
    } else {
      const n = segment.index + 1;
      const link = el("a", { class: "cite", href: `#source-${n}`, title: segment.citedText }, String(n));
      link.addEventListener("click", () => highlightSource(n));
      paragraphs.at(-1).append(link);
    }
  }
  answerText.replaceChildren(...paragraphs.filter((p) => p.childNodes.length));
}

function highlightSource(n) {
  for (const node of document.querySelectorAll(".source.highlight")) node.classList.remove("highlight");
  const target = $(`source-${n}`);
  if (!target) return;
  if (target.closest("details")) othersEl.open = true;
  target.classList.add("highlight");
}

function sourceCard(source) {
  const n = source.index + 1;
  const badges = [];
  if (source.not_yet_applicable) badges.push(el("span", { class: "badge" }, `Se aplica desde ${formatDate(source.application_date)}`));
  if (source.repealed) badges.push(el("span", { class: "badge" }, "Derogado"));
  return el(
    "li",
    { class: "source", id: `source-${n}` },
    el("div", { class: "source-head" }, el("span", { class: "source-num" }, `[${n}]`), el("span", { class: "source-cite" }, source.citation), ...badges),
    source.heading ? el("p", { class: "source-heading" }, source.heading) : null,
    el("p", { class: "source-text" }, source.text),
    el(
      "div",
      { class: "source-foot" },
      el("span", {}, `Consolidado a ${formatDate(source.consolidated_as_of)}`),
      el("a", { href: source.url, target: "_blank", rel: "noopener" }, "Ver en el BOE ↗"),
    ),
  );
}

function renderSources(citedIndexes) {
  const cited = new Set(citedIndexes);
  const citedSources = sources.filter((s) => cited.has(s.index));
  const others = sources.filter((s) => !cited.has(s.index));
  citedEl.replaceChildren(...citedSources.map(sourceCard));
  uncitedEl.replaceChildren(...others.map(sourceCard));
  othersEl.hidden = others.length === 0;
  sourcesEmpty.hidden = sources.length > 0;
}

function renderMeta(done) {
  const parts = [];
  if (done.model) parts.push(done.model);
  if (done.timings_ms?.total) parts.push(`${(done.timings_ms.total / 1000).toFixed(1)} s`);
  if (done.usage?.input_tokens) parts.push(`${done.usage.input_tokens} + ${done.usage.output_tokens} tokens`);
  if (done.cost_usd !== null && done.cost_usd !== undefined) parts.push(`${done.cost_usd.toFixed(4)} USD`);
  metaEl.textContent = parts.join(" · ");
}

// --- Streaming ---------------------------------------------------------------------------------
function handleEvent(type, data) {
  if (type === "sources") {
    sources = data.sources;
    renderSources([]);
    statusEl.textContent = "Redactando la respuesta…";
  } else if (type === "delta") {
    segments.push({ type: "text", text: data.text });
    renderAnswer();
  } else if (type === "citation") {
    segments.push({ type: "cite", index: data.citation.source_index, citedText: data.citation.cited_text });
    renderAnswer();
  } else if (type === "warning") {
    warningsEl.append(el("p", { class: "warning" }, data.message));
  } else if (type === "done") {
    renderAnswer(data.abstained);
    renderSources(data.abstained ? [] : data.cited_sources);
    if (data.abstained) {
      // Keep what was consulted visible, so the reader can see why nothing matched.
      othersEl.hidden = sources.length === 0;
    }
    renderMeta(data);
    statusEl.textContent = "";
  } else if (type === "error") {
    throw new Error(data.message);
  }
}

async function ask(text, normIds) {
  const response = await fetch("/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: text, norm_ids: normIds }),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Error ${response.status}`);
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let type = "message";
      const data = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event: ")) type = line.slice(7);
        else if (line.startsWith("data: ")) data.push(line.slice(6));
      }
      if (data.length) handleEvent(type, JSON.parse(data.join("\n")));
    }
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (text.length < 3) {
    statusEl.textContent = "Escriba una pregunta de al menos tres caracteres.";
    return;
  }
  const normIds = [...document.querySelectorAll('input[name="norm"]:checked')].map((input) => input.value);
  segments = [];
  sources = [];
  answerEl.hidden = false;
  answerText.replaceChildren();
  warningsEl.replaceChildren();
  metaEl.textContent = "";
  renderSources([]);
  submit.disabled = true;
  statusEl.textContent = "Buscando en la normativa…";
  try {
    await ask(text, normIds);
  } catch (error) {
    statusEl.textContent = "";
    answerText.replaceChildren(el("p", { class: "error" }, `No se ha podido responder: ${error.message}`));
  } finally {
    submit.disabled = false;
  }
});

for (const button of document.querySelectorAll(".example")) {
  button.addEventListener("click", () => {
    question.value = button.textContent;
    question.focus();
  });
}

question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) form.requestSubmit();
});

loadNorms();
