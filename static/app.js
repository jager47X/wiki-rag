const chat = document.getElementById("chat");
const form = document.getElementById("input-form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");

let busy = false;

// Auto-resize textarea
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 120) + "px";
});

// Submit on Enter (Shift+Enter for newline)
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.dispatchEvent(new Event("submit"));
  }
});

// Suggestion buttons
document.querySelectorAll(".suggestion").forEach((btn) => {
  btn.addEventListener("click", () => {
    input.value = btn.dataset.q;
    form.dispatchEvent(new Event("submit"));
  });
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || busy) return;

  // Remove welcome screen
  const welcome = chat.querySelector(".welcome");
  if (welcome) welcome.remove();

  appendMsg("user", text);
  input.value = "";
  input.style.height = "auto";
  busy = true;
  sendBtn.disabled = true;

  const statusEl = createStatusBubble();

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    if (!resp.ok) throw new Error(`Server error: ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep incomplete line

      let eventType = null;
      for (const line of lines) {
        if (line.startsWith("event: ")) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith("data: ")) {
          const data = JSON.parse(line.slice(6));
          if (eventType === "status") {
            updateStatus(statusEl, data.status);
          } else if (eventType === "done") {
            statusEl.remove();
            appendAssistant(data.answer, data.sources || []);
          }
          eventType = null;
        }
      }
    }
  } catch (err) {
    statusEl.remove();
    appendMsg("assistant", "Something went wrong. Please try again.");
    console.error(err);
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
});

function createStatusBubble() {
  const div = document.createElement("div");
  div.className = "msg assistant status-bubble";
  div.innerHTML = `
    <div class="status-dots"><span></span><span></span><span></span></div>
    <span class="status-text">Thinking...</span>
  `;
  chat.appendChild(div);
  scrollToBottom();
  return div;
}

function updateStatus(el, message) {
  const textEl = el.querySelector(".status-text");
  if (textEl) textEl.textContent = message;
  scrollToBottom();
}

function appendMsg(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chat.appendChild(div);
  scrollToBottom();
}

function appendAssistant(answer, sources) {
  const div = document.createElement("div");
  div.className = "msg assistant";

  const answerP = document.createElement("p");
  answerP.textContent = answer;
  div.appendChild(answerP);

  if (sources.length > 0) {
    const wrapper = document.createElement("div");
    wrapper.className = "sources";

    const toggle = document.createElement("button");
    toggle.className = "sources-toggle";
    toggle.textContent = `${sources.length} source${sources.length > 1 ? "s" : ""}`;
    toggle.addEventListener("click", () => {
      toggle.classList.toggle("open");
      list.classList.toggle("open");
    });

    const list = document.createElement("div");
    list.className = "sources-list";

    sources.forEach((s) => {
      const card = document.createElement("div");
      card.className = "source-card";
      card.innerHTML = `
        <span class="score">${s.score}</span>
        <span class="title">${esc(s.title)}</span>
        ${s.section ? `<span class="section"> — ${esc(s.section)}</span>` : ""}
        <div class="excerpt">${esc(s.text.slice(0, 200))}${s.text.length > 200 ? "…" : ""}</div>
      `;
      list.appendChild(card);
    });

    wrapper.appendChild(toggle);
    wrapper.appendChild(list);
    div.appendChild(wrapper);
  }

  chat.appendChild(div);
  scrollToBottom();
}

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function esc(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}
