// หน้ากากของ service -- ไม่มี logic การอัดอยู่ที่นี่เลย ทุกอย่างเป็นการอ่าน
// /api/state แล้ววาด กับการยิงคำสั่งสองตัว (เปิดห้อง / ปิดห้อง)
//
// ข้อความของ "เหตุการณ์" มาจาก service แล้ว (field .text) เพราะ catalog อยู่ที่
// src/messages.py ที่เดียว ส่วน UI ที่นี่มีแค่ป้ายกำกับของหน้าจอเอง

const UI = {
  th: {
    title: "ตัวรันบันทึกประชุม",
    worker: "ตัวประมวลผลพร้อม",
    workerOff: "ตัวประมวลผลไม่พร้อม",
    workerOffNote:
      "ยังอัดได้ตามปกติ ไฟล์จะเข้าคิวรอไว้ แล้วประมวลผลเมื่อตัวประมวลผลกลับมา",
    offline: "ติดต่อตัวรันไม่ได้ — ตรวจว่าหน้าต่าง MeetingRunnerUI ยังเปิดอยู่",
    mode: "รูปแบบการใช้ AI",
    room: "ชื่อห้อง (ไม่ใส่ก็ได้)",
    roomPh: "เช่น standup เช้าวันจันทร์",
    open: "เปิดห้อง",
    rec: "กำลังอัด",
    close: "ปิดการประชุม",
    mic: "ไมค์",
    spk: "ลำโพง",
    closing: "กำลังปิด…",
    untitled: "ประชุมไม่ได้ตั้งชื่อ",
    activity: "จอแสดงผลการทำงาน",
    cfTitle: "ปิดการประชุมใช่ไหม",
    cfBody: "หยุดอัดแล้วส่งเข้าประมวลผลทันที ประชุมนี้อัดซ้ำไม่ได้",
    cfNo: "อัดต่อ",
    cfYes: "ปิดเลย",
    steps: ["บีบอัดไฟล์เสียง", "ถอดเสียง", "แยกผู้พูด", "สรุป", "เสร็จ"],
    doneTitle: "บันทึกเรียบร้อย",
    again: "เปิดห้องใหม่",
    failed: "ประมวลผลไม่สำเร็จ ดูรายละเอียดในจอแสดงผลการทำงาน",
    lang: "EN",
    models: [
      ["GLM-5.2", "GLM 5.2", "ข้อมูลไม่ออกนอกบริษัท · ช้ากว่า"],
      ["claude-opus-5", "Opus 5", "แม่นสุด · $5/$25 ต่อ MTok"],
      ["claude-sonnet-5", "Sonnet 5", "ประหยัด · $3/$15 ต่อ MTok"],
      ["transcript-only", "ถอดเสียงอย่างเดียว", "ไม่สรุป · ไม่เสียเงิน"],
    ],
  },
  en: {
    title: "Meeting recorder",
    worker: "Worker ready",
    workerOff: "Worker not running",
    workerOffNote:
      "You can still record. The file waits in the queue and is processed when the worker comes back.",
    offline: "Cannot reach the runner — check that the MeetingRunnerUI window is still open",
    mode: "AI mode",
    room: "Room name (optional)",
    roomPh: "e.g. monday standup",
    open: "Open room",
    rec: "Recording",
    close: "End meeting",
    mic: "Mic",
    spk: "Speaker",
    closing: "Ending…",
    untitled: "Untitled meeting",
    activity: "Activity",
    cfTitle: "End this meeting?",
    cfBody: "Recording stops and processing starts. A meeting cannot be re-recorded.",
    cfNo: "Keep recording",
    cfYes: "End it",
    steps: ["Encoding audio", "Transcribing", "Separating speakers", "Summarizing", "Done"],
    doneTitle: "Saved",
    again: "Open another room",
    failed: "Processing failed — see the activity panel for details",
    lang: "TH",
    models: [
      ["GLM-5.2", "GLM 5.2", "Stays in-house · slower"],
      ["claude-opus-5", "Opus 5", "Most accurate · $5/$25 per MTok"],
      ["claude-sonnet-5", "Sonnet 5", "Cheaper · $3/$15 per MTok"],
      ["transcript-only", "Transcript only", "No summary · no cost"],
    ],
  },
};

const NO_SUMMARY_MODEL = "transcript-only";
const SUMMARIZE_STEP = 3;

// ขั้นที่เหตุการณ์นี้พาไปถึง -- ชี้จากเหตุการณ์ล่าสุด ไม่นับสะสมเอง เพราะการ
// นับสะสมจะให้ตัวเลขผิดทันทีที่รีเฟรชหน้าจอกลางคัน
const STAGE_OF = {
  encode_started: 0,
  queued: 1,
  transcribe_started: 1,
  diarize_started: 2,
  summarize_started: 3,
  meeting_done: 4,
};

// อ่านภาษาก่อน render ครั้งแรกเสมอ ไม่งั้นหน้าจอจะกระพริบสลับภาษาตอนโหลด
let lang = localStorage.getItem("runnerLang") === "en" ? "en" : "th";
let model = "GLM-5.2";
let roomDraft = "";
let stopping = false;
let offline = false;
let lastState = null;
// เก็บไว้เอง เพราะ last_result ของ service ถูกล้างเมื่อเปิดห้องถัดไป
let followingJob = null;
let dismissedJob = null;

const el = (id) => document.getElementById(id);
const t = () => UI[lang];
const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function fmtClock(sec) {
  const h = String(Math.floor(sec / 3600)).padStart(2, "0");
  const m = String(Math.floor(sec / 60) % 60).padStart(2, "0");
  const s = String(Math.floor(sec) % 60).padStart(2, "0");
  return `${h}:${m}:${s}`;
}

function jobStemOf(path) {
  if (!path) return null;
  const base = String(path).split(/[\\/]/).pop();
  return base.replace(/\.[^.]+$/, "");
}

function warningsHtml(state) {
  if (!state || !state.warnings || !state.warnings.length) return "";
  return state.warnings
    .map((w) => `<div class="note warn">⚠ ${esc(w.text || w.code)}</div>`)
    .join("");
}

function viewIdle(state) {
  const x = t();
  const options = x.models
    .map(
      ([id, title, desc]) => `
      <div class="opt ${model === id ? "on" : ""}" data-model="${id}">
        <span class="tick">✓</span>
        <span><span class="t">${esc(title)}</span><br><span class="d">${esc(desc)}</span></span>
      </div>`
    )
    .join("");
  // จุดสถานะดับไม่ปิดปุ่ม: การอัดไม่ได้พึ่ง GPU เลย ไฟล์รอในคิวได้ ถ้าบล็อกตรงนี้
  // เท่ากับทำให้พลาดประชุมด้วยเหตุผลที่รอทีหลังได้
  const workerNote =
    state && state.worker_ready === false
      ? `<div class="note warn">⚠ ${esc(x.workerOffNote)}</div>`
      : "";
  return `${warningsHtml(state)}${workerNote}
    <label class="field">${esc(x.mode)}</label>
    ${options}
    <label class="field" style="margin-top:14px">${esc(x.room)}</label>
    <input type="text" id="room" placeholder="${esc(x.roomPh)}" value="${esc(roomDraft)}" />
    <button class="primary" id="go" style="margin-top:14px">${esc(x.open)}</button>`;
}

function viewRecording(state) {
  const x = t();
  const isStopping = state.recorder === "stopping" || stopping;
  const modelLabel = (x.models.find((m) => m[0] === state.model) || [, state.model])[1];
  const devices = state.devices || {};
  const deviceRows =
    devices.mic || devices.loopback
      ? `<div class="devs">
           <div>${esc(x.mic)} · ${esc(devices.mic || "—")}</div>
           <div>${esc(x.spk)} · ${esc(devices.loopback || "—")}</div>
         </div>`
      : "";
  return `${warningsHtml(state)}
    <div class="rec-wrap">
      <span class="rec-badge"><span class="dot"></span>${esc(x.rec)}</span>
      <div class="timer" id="clock">${fmtClock(state.elapsed_seconds || 0)}</div>
      <div class="sub">${esc(state.room || x.untitled)} · ${esc(modelLabel || "")}</div>
      ${deviceRows}
      <button class="danger" id="stop" ${isStopping ? "disabled" : ""}>
        ${esc(isStopping ? x.closing : x.close)}
      </button>
    </div>`;
}

function viewProcessing(state, stage, failed) {
  const x = t();
  const skipSummary = state.model === NO_SUMMARY_MODEL;
  const rows = x.steps
    .map((label, i) => {
      if (skipSummary && i === SUMMARIZE_STEP) return "";
      const cls = i < stage ? "done" : i === stage ? "now" : "wait";
      const icon = i < stage ? "✓" : i === stage ? '<span class="spin">◠</span>' : "○";
      return `<div class="step ${cls}"><span class="ic">${icon}</span><span>${esc(label)}</span></div>`;
    })
    .join("");
  const failedNote = failed ? `<div class="note warn">⚠ ${esc(x.failed)}</div>` : "";
  return `${warningsHtml(state)}${failedNote}
    <div class="sub" style="text-align:center">${esc(followingJob || "")}</div>
    ${rows}`;
}

function viewDone(state) {
  const x = t();
  return `${warningsHtml(state)}
    <div class="rec-wrap">
      <div style="font-size:28px;color:var(--ok);line-height:1.2">✓</div>
      <div style="font-size:15px;font-weight:500;margin-top:2px">${esc(x.doneTitle)}</div>
      <div class="sub" style="font-family:Consolas,monospace;font-size:11.5px">
        ${esc(followingJob || "")}
      </div>
      <button class="primary" id="again">${esc(x.again)}</button>
    </div>`;
}

function jobProgress(state) {
  if (!followingJob) return null;
  const events = (state.activity || []).filter((e) => e.job === followingJob);
  if (!events.length) return { stage: 0, failed: false };
  let stage = 0;
  let failed = false;
  for (const e of events) {
    if (e.code === "job_failed") failed = true;
    if (e.code in STAGE_OF) stage = STAGE_OF[e.code];
  }
  return { stage, failed };
}

let lastSignature = null;

// วาดใหม่ทั้งก้อนเฉพาะตอนสถานะเปลี่ยนจริง ไม่ใช่ทุกครั้งที่ poll -- การแทน
// innerHTML ทุกวินาทีจะดีดเคอร์เซอร์ออกจากช่องชื่อห้องระหว่างที่ผู้ใช้พิมพ์อยู่
function signatureOf(state, view, progress) {
  if (!state) return "offline";
  return [
    view,
    state.recorder,
    state.worker_ready,
    state.room,
    state.model,
    JSON.stringify(state.devices || {}),
    model,
    lang,
    stopping,
    progress ? `${progress.stage}:${progress.failed}` : "",
    (state.warnings || []).map((w) => w.code).join(","),
  ].join("|");
}

function render(state) {
  const x = t();
  el("hTitle").textContent = x.title;
  el("langBtn").textContent = x.lang;
  el("footLabel").textContent = x.activity;
  el("cfTitle").textContent = x.cfTitle;
  el("cfBody").textContent = x.cfBody;
  el("cfNo").textContent = x.cfNo;
  el("cfYes").textContent = x.cfYes;

  if (offline || !state) {
    el("wDot").className = "dot off";
    el("wText").textContent = "";
    if (lastSignature !== "offline") {
      el("body").innerHTML = `<div class="note warn">⚠ ${esc(x.offline)}</div>`;
      lastSignature = "offline";
    }
    return;
  }

  el("wDot").className = "dot" + (state.worker_ready ? "" : " off");
  el("wText").textContent = state.worker_ready ? x.worker : x.workerOff;

  const recording =
    state.recorder === "recording" || state.recorder === "stopping";
  const progress = recording ? null : jobProgress(state);
  let view = "idle";
  if (recording) view = "recording";
  else if (progress && followingJob !== dismissedJob)
    view = progress.stage >= 4 ? "done" : "processing";

  const signature = signatureOf(state, view, progress);
  if (signature !== lastSignature) {
    el("body").innerHTML =
      view === "recording"
        ? viewRecording(state)
        : view === "done"
        ? viewDone(state)
        : view === "processing"
        ? viewProcessing(state, progress.stage, progress.failed)
        : viewIdle(state);
    lastSignature = signature;
    wire();
  }

  // นาฬิกาเดินทุกวินาทีโดยไม่ต้องวาดใหม่ทั้งก้อน
  const clock = el("clock");
  if (clock) clock.textContent = fmtClock(state.elapsed_seconds || 0);
  renderLog(state);
}

function renderLog(state) {
  el("log").innerHTML = (state.activity || [])
    .slice(-60)
    .map((e) => {
      const time = String(e.ts || "").slice(11, 19);
      return `<div class="${esc(e.level || "info")}">${esc(time)}  ${esc(e.text || e.code)}</div>`;
    })
    .join("");
}

function wire() {
  document.querySelectorAll(".opt").forEach((o) => {
    o.onclick = () => {
      model = o.dataset.model;
      render(lastState);
    };
  });
  const room = el("room");
  if (room) room.oninput = () => (roomDraft = room.value);
  const go = el("go");
  if (go) go.onclick = openRoom;
  const stop = el("stop");
  if (stop) stop.onclick = () => el("scrim").classList.remove("hide");
  const again = el("again");
  if (again)
    again.onclick = () => {
      dismissedJob = followingJob;
      render(lastState);
    };
}

async function openRoom() {
  const room = el("room");
  roomDraft = room ? room.value.trim() : "";
  try {
    const response = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, name: roomDraft }),
    });
    // 409 = มีห้องเปิดอยู่แล้ว ให้สถานะจริงชนะ ไม่ต้องเดา
    if (response.status === 201) {
      followingJob = null;
      dismissedJob = null;
      roomDraft = "";
    }
  } catch (e) {
    offline = true;
  }
  await poll();
}

async function stopRoom() {
  stopping = true;
  try {
    await fetch("/api/session/stop", { method: "POST" });
  } catch (e) {
    offline = true;
  }
  await poll();
}

async function poll() {
  let state;
  try {
    const response = await fetch(`/api/state?lang=${encodeURIComponent(lang)}`);
    state = await response.json();
    offline = false;
  } catch (e) {
    offline = true;
    render(null);
    return;
  }
  if (state.recorder === "recording") stopping = false;
  // เริ่มตามงานที่เพิ่งอัดเสร็จ ตั้งแต่วินาทีที่ service บอกว่าได้ไฟล์แล้ว
  if (state.last_result) {
    const stem = jobStemOf(state.last_result);
    if (stem && stem !== followingJob && stem !== dismissedJob) followingJob = stem;
  }
  lastState = state;
  render(state);
}

el("langBtn").onclick = () => {
  lang = lang === "th" ? "en" : "th";
  localStorage.setItem("runnerLang", lang);
  poll();
};
el("cfNo").onclick = () => el("scrim").classList.add("hide");
el("cfYes").onclick = () => {
  el("scrim").classList.add("hide");
  stopRoom();
};

render(null);
poll();
setInterval(poll, 1000);
