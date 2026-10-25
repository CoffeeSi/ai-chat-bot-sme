"use strict";

const $ = (selector) => document.querySelector(selector);
const STORE = "dameli.care.chats.v1";
const SESSION_KEY = "dameli.care.api-key";
const MAX_CHATS = 20;
const MAX_MESSAGES = 60;
const state = { chats: [], current: null, products: [], online: false, busy: false, pending: null, connectVersion: 0 };
let toastTimer;

function node(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined) result.textContent = String(text);
  return result;
}

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(svg.namespaceURI, "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

function storageGet(store, key) {
  try { return store.getItem(key); } catch { return null; }
}

function toast(message) {
  clearTimeout(toastTimer);
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  toastTimer = setTimeout(() => { $("#toast").hidden = true; }, 3800);
}

function currentChat() { return state.chats.find((chat) => chat.id === state.current); }

function save() {
  state.chats = state.chats.slice(0, MAX_CHATS);
  try {
    localStorage.setItem(STORE, JSON.stringify({ current: state.current, chats: state.chats }));
  } catch { /* A private browser may disable persistence; the current chat still works. */ }
}

function restore() {
  try {
    const stored = JSON.parse(storageGet(localStorage, STORE) || "null");
    if (!stored || !Array.isArray(stored.chats)) return;
    state.chats = stored.chats.filter((chat) => typeof chat.id === "string" && Array.isArray(chat.messages)).slice(0, MAX_CHATS);
    for (const chat of state.chats) {
      chat.messages = chat.messages.filter((message) => ["user", "assistant", "error"].includes(message.role) && typeof message.text === "string").slice(-MAX_MESSAGES);
      if (!chat.product || typeof chat.product.item_id !== "string") chat.product = null;
    }
    state.current = state.chats.some((chat) => chat.id === stored.current) ? stored.current : state.chats[0]?.id;
  } catch { state.chats = []; }
}

function closeSidebar() {
  $("#sidebar").classList.remove("is-open");
  $("#sidebar-backdrop").hidden = true;
  $("#mobile-menu").setAttribute("aria-expanded", "false");
  syncSidebarAccessibility();
}

function syncSidebarAccessibility() {
  $("#sidebar").inert = window.matchMedia("(max-width: 680px)").matches && !$("#sidebar").classList.contains("is-open");
}

function newChat() {
  if (state.busy) return toast("Сначала дождитесь ответа или остановите запрос.");
  const empty = state.chats.find((chat) => !chat.messages.length);
  const chat = empty || { id: crypto.randomUUID(), title: "Новый диалог", created: Date.now(), messages: [], product: null };
  if (!empty) state.chats.unshift(chat);
  state.current = chat.id;
  $("#question").value = "";
  resizeComposer();
  save();
  renderAll();
  closeSidebar();
  $("#question").focus();
}

function renderHistory() {
  const history = $("#history");
  history.replaceChildren();
  const chats = state.chats.filter((chat) => chat.messages.length);
  $("#history-count").textContent = chats.length;
  if (!chats.length) history.append(node("p", "history-empty", "Здесь появятся ваши диалоги. Начнём с первого вопроса?"));
  for (const chat of chats) {
    const button = node("button", `history-item${chat.id === state.current ? " active" : ""}`);
    button.title = chat.title;
    button.setAttribute("aria-current", chat.id === state.current ? "true" : "false");
    const label = node("span");
    label.append(node("span", "history-title", chat.title), node("span", "history-date", new Date(chat.created).toLocaleDateString("ru-RU", { day: "numeric", month: "short" })));
    button.append(icon("chat"), label);
    button.addEventListener("click", () => {
      if (state.busy) return toast("Сначала дождитесь ответа или остановите запрос.");
      state.current = chat.id;
      save();
      renderAll();
      closeSidebar();
    });
    history.append(button);
  }
}

function productUrl(sku) { return `https://ozon.kz/product/${encodeURIComponent(sku)}/`; }

function renderProduct() {
  const product = currentChat()?.product;
  $("#selected-product").hidden = !product;
  $("#empty-product").hidden = Boolean(product);
  $("#context-strict").hidden = !product;
  $("#picker-label").textContent = product ? "Изменить товар" : "Выбрать товар";
  const context = $("#context-product");
  context.replaceChildren(document.createTextNode(product?.name || (product ? product.item_id : "Все товары")), icon("chevron"));
  context.title = product?.name || "Выбрать товар";
  if (product) {
    $("#product-name").textContent = product.name || "Товар без названия";
    $("#product-sku").textContent = `Артикул ${product.item_id}`;
    $("#product-link").href = productUrl(product.item_id);
  }
}

function inlineText(parent, text) {
  // A small text-only formatter: neither model output nor history is inserted as HTML.
  for (const part of text.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`)/g)) {
    if (part.startsWith("**") && part.endsWith("**")) parent.append(node("strong", "", part.slice(2, -2)));
    else if (part.startsWith("`") && part.endsWith("`")) parent.append(node("code", "", part.slice(1, -1)));
    else parent.append(document.createTextNode(part));
  }
}

function answerContent(text) {
  const content = node("div", "answer-body");
  for (const line of text.split("\n")) {
    if (!line.trim()) { content.append(node("div", "line-gap")); continue; }
    const heading = /^#{1,4}\s/.test(line);
    const block = node("div", `answer-line${heading ? " heading" : ""}`);
    inlineText(block, line.replace(/^#{1,4}\s+/, "").replace(/^\s*[-*]\s+/, "• "));
    content.append(block);
  }
  return content;
}

function sourceList(sources) {
  const details = node("details", "sources");
  const summary = node("summary");
  summary.append(icon("book"), node("span", "", `Источники ответа · ${sources.length}`), icon("chevron"));
  details.append(summary);
  for (const source of sources) {
    const item = node("div", "source-item");
    const title = node(source.item_id ? "a" : "span", "source-title", source.product || "Информация о товаре");
    if (source.item_id) {
      title.href = productUrl(source.item_id);
      title.target = "_blank";
      title.rel = "noopener noreferrer";
      title.append(icon("link"));
    }
    item.append(title, node("span", "source-kind", source.source === "product" ? source.question || "Карточка товара" : "Вопросы и ответы покупателей"));
    if (source.same_item === false) item.append(node("p", "source-mismatch", "Этот источник относится к другому товару"));
    const answers = Array.isArray(source.answers) ? source.answers.map((answer) => answer.text).filter(Boolean).join("\n") : "";
    const preview = source.content || [source.question, answers].filter(Boolean).join("\n");
    if (preview) item.append(node("p", "source-preview", preview));
    details.append(item);
  }
  return details;
}

function renderMessage(message) {
  const row = node("article", `message ${message.role}`);
  if (message.role === "user") {
    const meta = node("div", "message-meta", new Date(message.at).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" }));
    meta.append(icon("check"));
    row.append(node("div", "user-bubble", message.text), meta);
    return row;
  }
  const label = node("div", "assistant-label");
  const avatar = node("span", "assistant-avatar");
  avatar.append(icon("spark"));
  label.append(avatar, node("span", "", "Dámeli Care"), node("small", "", message.model || "Помощник магазина"));
  row.append(label);
  if (message.role === "error") {
    const error = node("div", "error-message", message.text);
    if (message.retry) {
      const retry = node("button", "retry-button", "Повторить запрос");
      retry.addEventListener("click", () => { if (!state.busy) sendQuestion(message.retry); });
      error.append(retry);
    }
    row.append(error);
    return row;
  }
  row.append(answerContent(message.text));
  if (message.answered === false) row.append(node("span", "no-answer", "В базе пока недостаточно информации по этому вопросу"));
  if (Array.isArray(message.sources) && message.sources.length) row.append(sourceList(message.sources));
  const actions = node("div", "answer-actions");
  const copy = node("button", "copy-answer");
  copy.append(icon("copy"), document.createTextNode("Скопировать"));
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(message.text); toast("Ответ скопирован"); }
    catch { toast("Не удалось скопировать. Выделите текст ответа вручную."); }
  });
  actions.append(copy);
  row.append(actions);
  return row;
}

function renderMessages() {
  const messages = currentChat()?.messages || [];
  $("#welcome").hidden = messages.length > 0;
  $("#messages").replaceChildren(...messages.map(renderMessage));
  if (state.busy) {
    const pending = node("div", "message pending");
    const typing = node("div", "typing");
    const dots = node("span", "typing-dots");
    dots.append(node("i"), node("i"), node("i"));
    typing.append(dots, node("span", "", "Ищем информацию и готовим ответ…"));
    pending.append(typing);
    $("#messages").append(pending);
  }
  requestAnimationFrame(() => {
    const area = $("#chat-scroll");
    area.scrollTop = messages.length ? area.scrollHeight : 0;
  });
}

function renderAll() { renderHistory(); renderProduct(); renderMessages(); updateSend(); }

function updateSend() {
  const button = $("#send");
  button.disabled = !state.busy && (!state.online || $("#question").value.trim().length < 2);
  button.setAttribute("aria-label", state.busy ? "Остановить ожидание ответа" : "Отправить сообщение");
  button.title = state.busy ? "Остановить ожидание" : "Отправить";
  button.replaceChildren(icon(state.busy ? "stop" : "arrow"));
  $("#question").readOnly = state.busy;
}

function resizeComposer() {
  const input = $("#question");
  input.style.height = "auto";
  input.style.height = `${Math.min(155, input.scrollHeight)}px`;
  updateSend();
}

class RequestError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const apiKey = storageGet(sessionStorage, SESSION_KEY);
  if (apiKey) headers.set("X-API-Key", apiKey);
  if (options.body) headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...options, headers });
  let data;
  try { data = await response.json(); } catch { throw new RequestError("Сервер вернул неожиданный ответ. Попробуйте обновить страницу.", response.status); }
  if (!response.ok) {
    let message = typeof data.detail === "string" ? data.detail : "Не удалось выполнить запрос. Попробуйте ещё раз.";
    if (response.status === 401) message = "Нужен ключ доступа к поддержке. Укажите его в настройках подключения.";
    if (response.status === 429) message = "Слишком много вопросов подряд. Подождите минуту и попробуйте ещё раз.";
    if (response.status === 502) message = "Помощник временно недоступен. Ваш вопрос сохранён — попробуйте повторить запрос.";
    if (response.status === 503) message = "Gemini сейчас перегружена. Подождите немного и повторите вопрос.";
    throw new RequestError(message, response.status);
  }
  return data;
}

async function connect() {
  const version = ++state.connectVersion;
  state.online = false;
  $("#connection").className = "connection";
  $("#connection-text").textContent = "Подключаемся";
  updateSend();
  try {
    const signal = AbortSignal.timeout(15000);
    const [health, products] = await Promise.all([api("/health", { signal }), api("/products", { signal })]);
    if (version !== state.connectVersion) return;
    if (!Array.isArray(products)) throw new Error("Некорректный список товаров");
    state.products = products;
    state.online = true;
    $("#connection").className = "connection online";
    $("#connection-text").textContent = "На связи";
    $("#knowledge-stats").textContent = `${products.length} товаров · ${health.records} записей`;
    renderProductResults();
  } catch (error) {
    if (version !== state.connectVersion) return;
    state.products = [];
    $("#connection").className = "connection offline";
    $("#connection-text").textContent = error.status === 401 ? "Нужен ключ" : "Нет подключения";
    $("#knowledge-stats").textContent = "База знаний недоступна";
    renderProductResults();
    toast(error instanceof RequestError ? error.message : "Нет связи с сервером. Нажмите на статус подключения, чтобы повторить.");
    if (error.status === 401) openSettings();
  }
  updateSend();
}

async function sendQuestion(text) {
  if (state.busy || !state.online || text.trim().length < 2) return;
  const chat = currentChat();
  if (chat.messages.length >= MAX_MESSAGES - 1) return toast("В этом диалоге уже много сообщений. Начните новый диалог.");
  const question = text.trim().slice(0, 1000);
  chat.messages.push({ role: "user", text: question, at: Date.now() });
  if (chat.title === "Новый диалог") chat.title = question.slice(0, 60);
  state.chats = [chat, ...state.chats.filter((entry) => entry.id !== chat.id)];
  const controller = new AbortController();
  state.pending = controller;
  state.busy = true;
  $("#question").value = "";
  resizeComposer();
  save();
  renderAll();
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 180000);
  try {
    const data = await api("/ask", {
      method: "POST", signal: controller.signal,
      body: JSON.stringify({ question, item: chat.product?.item_id || null, strict: Boolean(chat.product) }),
    });
    if (typeof data.answer !== "string") throw new Error("Некорректный ответ сервера");
    chat.messages.push({ role: "assistant", text: data.answer, model: data.llm_model, sources: data.sources || [], answered: data.answered, at: Date.now() });
  } catch (error) {
    const aborted = controller.signal.aborted;
    const message = timedOut ? "Ответ занимает больше времени, чем обычно. Попробуйте повторить запрос." : aborted ? "Ожидание остановлено. Можно повторить вопрос." : error instanceof RequestError ? error.message : "Не удалось связаться с сервером. Проверьте подключение и повторите вопрос.";
    chat.messages.push({ role: "error", text: message, retry: question, at: Date.now() });
    if (error.status === 401) openSettings();
  } finally {
    clearTimeout(timeout);
    state.pending = null;
    state.busy = false;
    save();
    renderAll();
    $("#question").focus();
  }
}

function selectProduct(product) {
  if (state.busy) return toast("Дождитесь ответа, прежде чем менять товар.");
  currentChat().product = product;
  save();
  renderProduct();
  $("#product-dialog").close();
  $("#question").focus();
}

function renderProductResults() {
  const search = $("#product-search").value.trim().toLocaleLowerCase();
  const results = state.products.filter((product) => `${product.name || ""} ${product.item_id}`.toLocaleLowerCase().includes(search));
  const list = $("#product-results");
  list.replaceChildren();
  if (!search) {
    const all = node("button", "product-option");
    const label = node("span", "", "Весь ассортимент");
    label.append(node("small", "", "Искать по всем товарам магазина"));
    all.append(icon("search"), label);
    all.addEventListener("click", () => selectProduct(null));
    list.append(all);
  }
  for (const product of results.slice(0, 150)) {
    const option = node("button", "product-option");
    const box = node("span", "picker-icon"); box.append(icon("box"));
    const label = node("span", "", product.name || "Товар без названия");
    label.append(node("small", "", `Артикул ${product.item_id}`));
    option.append(box, label);
    if (currentChat()?.product?.item_id === product.item_id) option.append(icon("check"));
    option.addEventListener("click", () => selectProduct({ item_id: product.item_id, name: product.name }));
    list.append(option);
  }
  if (!results.length) list.append(node("p", "results-empty", state.online ? "Товаров не найдено. Попробуйте другое название или артикул." : "Сначала подключитесь к серверу."));
  $("#product-count").textContent = `Найдено товаров: ${results.length}${results.length > 150 ? " · Уточните запрос, чтобы увидеть остальные" : ""}`;
}

function openProducts() {
  if (state.busy) return toast("Дождитесь ответа, прежде чем менять товар.");
  $("#product-search").value = "";
  renderProductResults();
  $("#product-dialog").showModal();
  $("#product-search").focus();
}

function openSettings() {
  $("#api-key").value = storageGet(sessionStorage, SESSION_KEY) || "";
  if (!$("#settings-dialog").open) $("#settings-dialog").showModal();
}

$("#new-chat").addEventListener("click", newChat);
$("#mobile-menu").addEventListener("click", () => {
  $("#sidebar").classList.add("is-open");
  $("#sidebar-backdrop").hidden = false;
  $("#mobile-menu").setAttribute("aria-expanded", "true");
  syncSidebarAccessibility();
});
window.matchMedia("(max-width: 680px)").addEventListener("change", syncSidebarAccessibility);
$("#sidebar-backdrop").addEventListener("click", closeSidebar);
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeSidebar(); });
$("#product-picker").addEventListener("click", openProducts);
$("#context-product").addEventListener("click", openProducts);
$("#clear-product").addEventListener("click", () => selectProduct(null));
$("#product-search").addEventListener("input", renderProductResults);
$("#open-settings").addEventListener("click", openSettings);
$("#connection").addEventListener("click", () => { if (!state.busy) connect(); });
$("#question").addEventListener("input", resizeComposer);
$("#question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!state.busy) $("#chat-form").requestSubmit();
  }
});
$("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.busy) state.pending?.abort();
  else sendQuestion($("#question").value);
});
$("#suggestions").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-question]");
  if (!button) return;
  $("#question").value = button.dataset.question;
  resizeComposer();
  $("#question").focus();
  if (!currentChat()?.product) openProducts();
});
document.querySelectorAll("[data-close]").forEach((button) => button.addEventListener("click", () => document.getElementById(button.dataset.close).close()));
document.querySelectorAll("dialog").forEach((dialog) => dialog.addEventListener("click", (event) => {
  if (event.target !== dialog) return;
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
}));
$("#settings-form").addEventListener("submit", (event) => {
  event.preventDefault();
  try { sessionStorage.setItem(SESSION_KEY, $("#api-key").value.trim()); }
  catch { return toast("Браузер не разрешает сохранять ключ для этой вкладки."); }
  $("#settings-dialog").close();
  connect();
});
$("#delete-chats").addEventListener("click", () => {
  if (state.busy) return toast("Сначала остановите текущий запрос.");
  if (!window.confirm("Удалить все диалоги из этого браузера?")) return;
  state.chats = [];
  $("#settings-dialog").close();
  newChat();
  toast("История диалогов удалена");
});
restore();
syncSidebarAccessibility();
if (!state.chats.length) newChat();
else renderAll();
resizeComposer();
connect();
