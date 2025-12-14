/* ============================
      CONFIG + ELEMENTS
============================ */

const api = {
  upload: "/api/upload",
  remove_doc: "/remove_doc"
};

const elements = {
  form: document.getElementById("mainForm"),
  runBtn: document.getElementById("runBtn"),
  resultDot: document.getElementById("resultIndicator"),

  promptMicBtn: document.getElementById("promptMicBtn"),
  promptMicStatus: document.getElementById("promptMicStatus"),
  promptAudioField: document.getElementById("promptAudioField"),

  liveMicBtn: document.getElementById("liveMicBtn"),
  liveStatus: document.getElementById("liveStatus"),
  liveAudioField: document.getElementById("liveAudioField"),

  sideDocInput: document.getElementById("sideDocInput"),
  sideDocList: document.getElementById("sideDocList"),
  sideDocCount: document.getElementById("sideDocCount"),
  addSideDocs: document.getElementById("addSideDocs"),

  themeToggle: document.getElementById("themeToggle"),
  themeLabel: document.getElementById("themeLabel"),
};

const MAX_DOCS = 5;


/* ============================
      AUDIO RECORDERS
============================ */

let promptRecorder;
let promptChunks = [];
let liveRecorder;
let liveChunks = [];

async function getAudioStream() {
  return navigator.mediaDevices.getUserMedia({ audio: true });
}

async function ensurePromptRecorder() {
  if (promptRecorder) return;
  const stream = await getAudioStream();
  promptRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

  promptRecorder.ondataavailable = (e) => {
    if (e.data.size) promptChunks.push(e.data);
  };

  promptRecorder.onstop = () => {
    handleBlobUpload(
      promptChunks,
      "prompt_",
      elements.promptAudioField,
      elements.promptMicStatus
    );
  };
}

async function ensureLiveRecorder() {
  if (liveRecorder) return;
  const stream = await getAudioStream();
  liveRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

  liveRecorder.ondataavailable = (e) => {
    if (e.data.size) liveChunks.push(e.data);
  };

  liveRecorder.onstop = () => {
    handleBlobUpload(
      liveChunks,
      "live_",
      elements.liveAudioField,
      elements.liveStatus
    );
  };
}

async function handleBlobUpload(chunks, prefix, hiddenField, statusEl) {
  const blob = new Blob(chunks, { type: "audio/webm" });
  chunks.length = 0;

  const fd = new FormData();
  fd.append("file", blob, prefix + "clip.webm");

  if (statusEl) statusEl.textContent = "Uploading...";

  try {
    const res = await fetch(api.upload, { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok || !data.filename) throw new Error("upload failed");

    hiddenField.value = data.filename;
    statusEl.textContent = "Ready · will be used on next run.";
  } catch (err) {
    statusEl.textContent = "Upload failed. Try again.";
  }
}


/* ============================
       MIC BUTTON HANDLERS
============================ */

elements.promptMicBtn?.addEventListener("click", async () => {
  try {
    await ensurePromptRecorder();
  } catch (e) {
    elements.promptMicStatus.textContent = "Mic permission denied.";
    return;
  }

  if (promptRecorder.state === "inactive") {
    promptRecorder.start();
    elements.promptMicBtn.classList.add("active");
    elements.promptMicStatus.textContent =
      "Recording prompt… click again to stop.";
  } else {
    promptRecorder.stop();
    elements.promptMicBtn.classList.remove("active");
  }
});

elements.liveMicBtn?.addEventListener("click", async () => {
  try {
    await ensureLiveRecorder();
  } catch (e) {
    elements.liveStatus.textContent = "Mic permission denied.";
    return;
  }

  if (liveRecorder.state === "inactive") {
    liveRecorder.start();
    elements.liveMicBtn.classList.add("active");
    elements.liveStatus.textContent = "Recording… tap again to stop.";
  } else {
    liveRecorder.stop();
    elements.liveMicBtn.classList.remove("active");
    elements.liveStatus.textContent = "Processing recording…";
  }
});


/* ============================
       DOCUMENT SIDEBAR (FIXED)
============================ */

/**
 * serverDocs = names of files already uploaded and stored in Flask session
 * localDocs  = File objects newly selected this run
 */

let serverDocs = [];
let localDocs = [];

/* ----------------------------
   ON PAGE LOAD – READ SERVER DOCS
-----------------------------*/
window.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".side-doc-item.static").forEach((li) => {
    const name = li.querySelector(".side-doc-name")?.textContent.trim();
    if (name) serverDocs.push(name);
  });
  renderSidebar();
});

/* ----------------------------
      RENDER SIDEBAR
-----------------------------*/
function renderSidebar() {
  elements.sideDocList.innerHTML = "";

  // server docs
  serverDocs.forEach((name, idx) => {
    const li = document.createElement("li");
    li.className = "side-doc-item";

    li.innerHTML = `
      <div class="side-doc-main">
        <span class="dot"></span>
        <span class="side-doc-name">${name}</span>
      </div>
      <button class="side-doc-remove" data-type="server" data-index="${idx}">×</button>
    `;
    elements.sideDocList.appendChild(li);
  });

  // local docs
  localDocs.forEach((file, idx) => {
    const li = document.createElement("li");
    li.className = "side-doc-item";

    li.innerHTML = `
      <div class="side-doc-main">
        <span class="dot"></span>
        <span class="side-doc-name">${file.name}</span>
      </div>
      <button class="side-doc-remove" data-type="local" data-index="${idx}">×</button>
    `;
    elements.sideDocList.appendChild(li);
  });

  elements.sideDocCount.textContent = serverDocs.length + localDocs.length;
}

/* ----------------------------
      ADD NEW DOCS
-----------------------------*/
elements.addSideDocs?.addEventListener("click", () =>
  elements.sideDocInput.click()
);

elements.sideDocInput?.addEventListener("change", () => {
  const newFiles = Array.from(elements.sideDocInput.files);

  newFiles.forEach((f) => {
    if (serverDocs.length + localDocs.length < MAX_DOCS) {
      localDocs.push(f);
    }
  });

  renderSidebar();
});

/* ----------------------------
      REMOVE DOCS
-----------------------------*/
elements.sideDocList.addEventListener("click", async (e) => {
  if (!e.target.classList.contains("side-doc-remove")) return;

  const type = e.target.dataset.type;
  const idx = parseInt(e.target.dataset.index);

  if (type === "server") {
    await fetch(api.remove_doc, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: `index=${idx}`,
    });
    serverDocs.splice(idx, 1);
  }

  if (type === "local") {
    localDocs.splice(idx, 1);
  }

  renderSidebar();
});

/* ----------------------------
      BEFORE SUBMIT – ATTACH ONLY NEW FILES
-----------------------------*/
elements.form?.addEventListener("submit", () => {
  const dt = new DataTransfer();

  localDocs.forEach((f) => dt.items.add(f));

  elements.sideDocInput.files = dt.files;

  elements.runBtn.classList.add("loading");
  elements.resultDot.classList.add("live");
});


/* ============================
         THEME TOGGLE
============================ */

(function setupThemeToggle() {
  if (!elements.themeToggle || !elements.themeLabel) return;

  const stored = localStorage.getItem("theme");
  if (stored === "light" || stored === "dark") {
    document.documentElement.setAttribute("data-theme", stored);
    elements.themeLabel.textContent = stored === "dark" ? "Dark" : "Light";
  } else {
    document.documentElement.setAttribute("data-theme", "dark");
    elements.themeLabel.textContent = "Dark";
  }

  elements.themeToggle.addEventListener("click", () => {
    const current =
      document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    elements.themeLabel.textContent = next === "dark" ? "Dark" : "Light";
  });
})();
