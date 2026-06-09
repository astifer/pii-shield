// PII Shield frontend: loads the NER model on-device (transformers.js, WebGPU, int8),
// runs NER on each message, highlights detected PII in red and replaces it with [TAG]s,
// then POSTs only the obfuscated text to the /chat mock-LLM endpoint via htmx.

import {
  env,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3";
import {
  buildHighlighted,
  buildObfuscated,
  mergeSpans,
} from "./obfuscate.js";

// Load model weights from this app's own /static/models (not the HF Hub).
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = "/static/models/";

const MODEL_ID = "pii-ner-rubert";

// --- DOM refs ---------------------------------------------------------------
const loadingEl = document.getElementById("loading");
const loadingLabel = document.getElementById("loading-label");
const progressEl = document.getElementById("progress");
const progressPct = document.getElementById("progress-pct");
const readyInfo = document.getElementById("ready-info");
const deviceBadge = document.getElementById("device-badge");
const inferenceTime = document.getElementById("inference-time");
const chatLog = document.getElementById("chat-log");
const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

// Average per-file progress so the bar reflects the whole model download.
const fileProgress = new Map();

function onProgress(e) {
  if (e.status === "progress" && e.file) {
    fileProgress.set(e.file, e.progress ?? 0);
    const vals = [...fileProgress.values()];
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    progressEl.value = avg;
    progressPct.textContent = `${Math.round(avg)}%`;
  } else if (e.status === "initiate" && e.file) {
    fileProgress.set(e.file, 0);
    loadingLabel.textContent = `Downloading ${e.file}…`;
  }
}

const useWebGPU = "gpu" in navigator;
const device = useWebGPU ? "webgpu" : "wasm";

let ner;
(async () => {
  try {
    ner = await pipeline("token-classification", MODEL_ID, {
      device,
      dtype: "q8", // int8
      progress_callback: onProgress,
    });
    onReady();
  } catch (err) {
    loadingLabel.textContent = `Failed to load model: ${err.message}`;
    console.error(err);
  }
})();

function onReady() {
  loadingEl.hidden = true;
  readyInfo.hidden = false;
  deviceBadge.textContent = useWebGPU ? "WebGPU" : "WASM (no WebGPU)";
  deviceBadge.className = useWebGPU ? "badge gpu" : "badge cpu";
  input.disabled = false;
  sendBtn.disabled = false;
  input.focus();
}

function appendUserBubble(highlightedHtml) {
  const div = document.createElement("div");
  div.className = "msg user";
  div.innerHTML =
    '<span class="who">You</span>' +
    `<div class="bubble">${highlightedHtml}</div>`;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

// --- submit flow ------------------------------------------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || !ner) return;

  input.disabled = true;
  sendBtn.disabled = true;

  // Run NER on-device and time it.
  const t0 = performance.now();
  let tokens = [];
  try {
    tokens = await ner(text, { aggregation_strategy: "simple" });
  } catch (err) {
    console.error("Inference error:", err);
  }
  const ms = performance.now() - t0;
  inferenceTime.textContent = `Inference: ${ms.toFixed(1)} ms`;

  const spans = mergeSpans(tokens);
  appendUserBubble(buildHighlighted(text, spans));
  const obfuscated = buildObfuscated(text, spans);

  input.value = "";

  // htmx does the POST of the obfuscated text; reply is appended to #chat-log.
  htmx.ajax("POST", "/chat", {
    values: { message: obfuscated },
    target: "#chat-log",
    swap: "beforeend",
  });

  input.disabled = false;
  sendBtn.disabled = false;
  input.focus();
});
