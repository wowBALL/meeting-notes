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
    enrollLink: "ลงทะเบียนเสียง",
    mode: "รูปแบบการใช้ AI",
    room: "ชื่อห้อง (ไม่ใส่ก็ได้)",
    roomPh: "เช่น standup เช้าวันจันทร์",
    open: "เปิดห้อง",
    rec: "กำลังอัด",
    close: "ปิดการประชุม",
    mic: "ไมค์",
    spk: "ลำโพง",
    muteMic: "🎙 ปิดไมค์",
    unmuteMic: "🔇 เปิดไมค์กลับ",
    micMutedNote: "ไมค์ปิดอยู่ — เสียงจากไมค์จะไม่ถูกอัด เสียงคู่สนทนายังอัดตามปกติ",
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
    spkTitle: "ผู้พูดที่ยังไม่รู้จัก",
    spkNamePh: "ใครพูด?",
    spkSave: "บันทึก",
    spkSkip: "ข้าม",
    spkPlay: "▶ ฟังเสียง",
    spkStop: "■ หยุด",
    spkSpoke: "พูดรวม",
    spkMin: "นาที",
    spkSec: "วินาที",
    spkGuess: "โมเดลเดาว่า",
    spkNear: "เสียงใกล้เคียงกับ",
    spkConfirmError: "บันทึกไม่สำเร็จ ลองใหม่อีกครั้ง",
    // คิว/ตัวอย่างที่มาจากก่อนอัปเดตระบบจำเสียง (หรือถูกแก้มือ) ไม่มีป้ายพื้นที่เวกเตอร์เลย
    // -- ต้องบอกว่าให้อัดใหม่/รอวิเคราะห์ใหม่ ไม่ใช่พูดเหมือนระบบพัง (missing_embedding_model)
    spkErrMissingEmbeddingModel:
      "รายการนี้มาจากก่อนอัปเดตระบบจำเสียง ไม่มีข้อมูลพอจะบันทึกได้ -- ข้ามคนนี้ไปได้ " +
      "อัดประชุมครั้งหน้าจะจำได้ตามปกติ",
    models: [
      ["GLM-5.2", "GLM 5.2", "ข้อมูลไม่ออกนอกบริษัท · ช้ากว่า"],
      ["claude-opus-5", "Opus 5", "แม่นสุด · $5/$25 ต่อ MTok"],
      ["claude-sonnet-5", "Sonnet 5", "ประหยัด · $3/$15 ต่อ MTok"],
      ["transcript-only", "ถอดเสียงอย่างเดียว", "ไม่สรุป · ไม่เสียเงิน"],
    ],
    kind: "ประเภทประชุม",
    // id ต้องตรงกับชื่อไฟล์ใน prompts/profiles/ และกับ prompts.KNOWN_PROFILES
    profiles: [
      ["dev", "dev ล้วน", "ศัพท์เทคนิคตรงๆ"],
      ["cross", "Business + dev", 'แยก "ทำได้" ออกจาก "จะทำ" · ขยายศัพท์ให้คนนอกทีม'],
    ],
  },
  en: {
    title: "Meeting recorder",
    worker: "Worker ready",
    workerOff: "Worker not running",
    workerOffNote:
      "You can still record. The file waits in the queue and is processed when the worker comes back.",
    offline: "Cannot reach the runner — check that the MeetingRunnerUI window is still open",
    enrollLink: "Enroll voices",
    mode: "AI mode",
    room: "Room name (optional)",
    roomPh: "e.g. monday standup",
    open: "Open room",
    rec: "Recording",
    close: "End meeting",
    mic: "Mic",
    spk: "Speaker",
    muteMic: "🎙 Mute mic",
    unmuteMic: "🔇 Unmute mic",
    micMutedNote: "Mic is muted — mic audio is not being recorded. The far end still is.",
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
    spkTitle: "Speakers we don't recognize yet",
    spkNamePh: "Who is this?",
    spkSave: "Save",
    spkSkip: "Skip",
    spkPlay: "▶ Play",
    spkStop: "■ Stop",
    spkSpoke: "spoke for",
    spkMin: "min",
    spkSec: "sec",
    spkGuess: "the model guessed",
    spkNear: "voice is close to",
    spkConfirmError: "Could not save. Try again.",
    // Queue entries/samples from before the voice-recognition upgrade (or hand-edited)
    // carry no embedding-space stamp at all -- say "record again", not "system is broken"
    spkErrMissingEmbeddingModel:
      "This entry predates the voice-recognition upgrade and lacks what's needed to save " +
      "-- you can skip it, the next recording will be recognized normally",
    models: [
      ["GLM-5.2", "GLM 5.2", "Stays in-house · slower"],
      ["claude-opus-5", "Opus 5", "Most accurate · $5/$25 per MTok"],
      ["claude-sonnet-5", "Sonnet 5", "Cheaper · $3/$15 per MTok"],
      ["transcript-only", "Transcript only", "No summary · no cost"],
    ],
    kind: "Meeting type",
    profiles: [
      ["dev", "dev only", "Technical terms as-is"],
      [
        "cross",
        "Business + dev",
        'Separates "can be done" from "will be done" · expands jargon',
      ],
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
  summarize_progress: 3,
  meeting_done: 4,
};

// อ่านภาษาก่อน render ครั้งแรกเสมอ ไม่งั้นหน้าจอจะกระพริบสลับภาษาตอนโหลด
let lang = localStorage.getItem("runnerLang") === "en" ? "en" : "th";
let model = "GLM-5.2";
// dev เป็นค่าเริ่มต้นเพราะเป็น 3 ใน 4 ครั้งของสัปดาห์ และการเผลอเลือก cross ในประชุม
// dev ล้วนทำให้โมเดล qualify คำพูดปกติเกินจำเป็นจนสรุปอ่านแล้วอ้อมค้อม
let profile = "dev";
let roomDraft = "";
let stopping = false;
let offline = false;
let lastState = null;
// เก็บไว้เอง เพราะ last_result ของ service ถูกล้างเมื่อเปิดห้องถัดไป
let followingJob = null;
let dismissedJob = null;
// คิวตั้งชื่อดึงแยกจาก /api/state เพราะมันเปลี่ยนแค่ตอนจบประชุมกับตอนผู้ใช้กดยืนยัน
// ไม่ใช่ทุกวินาทีเหมือนสถานะการอัด
let pendingMeetings = [];
let pendingTick = 0;
// งานที่เราดึงคิวให้แล้วหลังเห็นสัญญาณ speakers_pending -- กันไม่ให้ดึงซ้ำทุกวินาที
// ตลอดเวลาที่เหตุการณ์นั้นยังค้างอยู่ใน activity log
let pendingSignalJob = null;
// ช่องชื่อที่พิมพ์ค้างไว้ เก็บนอก DOM ด้วยเหตุผลเดียวกับ roomDraft: การวาดใหม่
// ทั้งก้อนจะดีดสิ่งที่พิมพ์ไปแล้วทิ้ง
const nameDrafts = {};
// ผู้พูดที่กด save/skip แล้วเซิร์ฟเวอร์ตอบไม่ใช่ 2xx -- เก็บ key ไว้เพื่อขึ้นบรรทัด
// เตือนสั้น ๆ ที่การ์ดนั้น ไม่ใช่ป้ายรวมของทั้งหน้า เพราะสาเหตุ (400/404/500) ผูกกับ
// ผู้พูดคนนั้นคนเดียว ไม่ใช่ทั้งหน้าจอ
const speakerErrors = {};
let audioEl = null;
let playingKey = null;

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

// หน่วยเวลาต้องมาจาก catalog เหมือนป้ายอื่น ไม่ใช่ฝังไว้ในฟังก์ชัน -- ไม่งั้นโหมด EN
// จะได้ "spoke for 45 วินาที" ปนกันสองภาษา (เห็นกับตาตอนตรวจหน้าจอจริง)
function fmtSpoken(seconds) {
  const x = t();
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m} ${x.spkMin} ${s} ${x.spkSec}` : `${s} ${x.spkSec}`;
}

function speakerKey(meeting, label) {
  return `${meeting}|${label}`;
}

function pendingHtml() {
  if (!pendingMeetings.length) return "";
  const x = t();
  const cards = pendingMeetings
    .map((meeting) =>
      meeting.speakers
        .map((speaker) => {
          const key = speakerKey(meeting.meeting_dir, speaker.label);
          const draft =
            nameDrafts[key] !== undefined
              ? nameDrafts[key]
              : (speaker.guess && speaker.guess.name) || "";
          const quote = (speaker.samples || [])
            .map((sample) => `<div class="quote">“${esc(sample.text)}”</div>`)
            .join("");
          const guess = speaker.guess
            ? `<div class="hint">${esc(x.spkGuess)} “${esc(speaker.guess.name)}”${
                speaker.guess.evidence ? ` — ${esc(speaker.guess.evidence)}` : ""
              }</div>`
            : "";
          const near = speaker.suggested
            ? `<div class="hint">${esc(x.spkNear)} “${esc(speaker.suggested.name)}”</div>`
            : "";
          const playing = playingKey === key;
          // speakerErrors[key] เก็บรหัส error ของเซิร์ฟเวอร์ไว้ (ไม่ใช่แค่ true/false
          // เหมือนเดิม) เพื่อให้ missing_embedding_model ได้ข้อความของตัวเอง แทนที่จะ
          // ถูกกลืนเป็น spkConfirmError ทั่วไปที่บอกให้ "ลองใหม่" ทั้งที่ลองใหม่ไม่ช่วย
          const errorCode = speakerErrors[key];
          const errorText =
            errorCode === "missing_embedding_model"
              ? x.spkErrMissingEmbeddingModel
              : x.spkConfirmError;
          const error = errorCode
            ? `<div class="note warn">⚠ ${esc(errorText)}</div>`
            : "";
          return `<div class="spk" data-key="${esc(key)}">
            <div class="who">${esc(speaker.label)}</div>
            <div class="meta">${esc(meeting.meeting_dir)} · ${esc(x.spkSpoke)} ${esc(
            fmtSpoken(speaker.speaking_seconds || 0)
          )}</div>
            ${quote}${guess}${near}
            <button class="play" data-play="${esc(key)}">${esc(
            playing ? x.spkStop : x.spkPlay
          )}</button>
            <div class="row">
              <input type="text" data-name="${esc(key)}" placeholder="${esc(
            x.spkNamePh
          )}" value="${esc(draft)}" />
              <button class="plain" data-save="${esc(key)}">${esc(x.spkSave)}</button>
              <button class="plain" data-skip="${esc(key)}">${esc(x.spkSkip)}</button>
            </div>
            ${error}
          </div>`;
        })
        .join("")
    )
    .join("");
  return `<label class="field">${esc(x.spkTitle)}</label>${cards}
    <div style="height:14px"></div>`;
}

function speakerAt(key) {
  for (const meeting of pendingMeetings) {
    for (const speaker of meeting.speakers) {
      if (speakerKey(meeting.meeting_dir, speaker.label) === key) {
        return { meeting: meeting.meeting_dir, speaker };
      }
    }
  }
  return null;
}

function stopPlayback() {
  if (audioEl) {
    audioEl.pause();
    audioEl = null;
  }
  playingKey = null;
}

// เล่นเฉพาะช่วงที่คนนั้นพูด ควบคุมด้วย JS ไม่ใช่ media fragment "#t=" ซึ่งเบราว์เซอร์
// รองรับไม่ทั่ว -- ไฟล์ประชุมยาวเป็นชั่วโมง การเล่นจากต้นไฟล์คือการไม่ได้ใช้ปุ่มนี้เลย
function playSample(key) {
  const found = speakerAt(key);
  const sample = found && (found.speaker.samples || [])[0];
  if (!sample) return;
  stopPlayback();
  audioEl = new Audio(`/api/speakers/audio/${encodeURIComponent(found.meeting)}`);
  audioEl.currentTime = sample.start;
  audioEl.ontimeupdate = () => {
    if (audioEl && audioEl.currentTime >= sample.end) {
      stopPlayback();
      render(lastState);
    }
  };
  audioEl.onended = () => {
    stopPlayback();
    render(lastState);
  };
  audioEl.onerror = () => {
    stopPlayback();
    render(lastState);
  };
  // currentTime ก่อนโหลด metadata ไม่มีผลกับบางเบราว์เซอร์ ตั้งซ้ำเมื่อพร้อมจริง
  audioEl.onloadedmetadata = () => {
    if (audioEl) audioEl.currentTime = sample.start;
  };
  audioEl.play().catch(() => {
    stopPlayback();
    render(lastState);
  });
  playingKey = key;
  render(lastState);
}

async function refreshPending() {
  try {
    const response = await fetch("/api/speakers/pending");
    const body = await response.json();
    pendingMeetings = body.meetings || [];
  } catch (e) {
    pendingMeetings = [];
  }
  render(lastState);
}

// หา <input> ของผู้พูดคนนี้จาก data-name แทนการเดา selector -- ใช้ค่าที่ผู้ใช้เห็น
// อยู่ตรงหน้าจอจริง ๆ (ไม่ว่าจะเป็นชื่อที่โมเดลเดาให้ตอนวาดครั้งแรก หรือที่พิมพ์แก้)
// ไม่ใช่ nameDrafts ซึ่งว่างเปล่าจนกว่าจะมี oninput ยิงสักครั้ง
function findNameInput(key) {
  for (const input of document.querySelectorAll("[data-name]")) {
    if (input.dataset.name === key) return input;
  }
  return null;
}

async function confirmSpeaker(key, skip) {
  const found = speakerAt(key);
  if (!found) return;
  const payload = { meeting: found.meeting, label: found.speaker.label };
  if (skip) {
    payload.skip = true;
  } else {
    // ช่องชื่อถูก prefill ด้วยชื่อที่โมเดลเดาให้เป็นค่า value ตั้งแต่วาดครั้งแรก แต่
    // nameDrafts[key] มีค่าเฉพาะเมื่อผู้ใช้เคยพิมพ์ (oninput ยิงแล้ว) เท่านั้น -- กด
    // บันทึกทันทีโดยไม่แตะช่องเลยตอนเดาถูกอยู่แล้วต้องส่งชื่อที่เห็นบนจอ ไม่ใช่ส่งค่า
    // ว่างเพราะ nameDrafts ยังไม่เคยถูกเติม (nameDrafts ยังต้องอยู่ต่อ เพราะการวาดใหม่
    // ทั้งก้อนตอนคิวเปลี่ยนจะดีดสิ่งที่พิมพ์ค้างไว้ทิ้งถ้าไม่มีมันเก็บสำรอง)
    const input = findNameInput(key);
    const name = (input ? input.value : nameDrafts[key] || "").trim();
    if (!name) return;
    payload.name = name;
  }
  try {
    const response = await fetch("/api/speakers/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      // เซิร์ฟเวอร์แยกสาเหตุไว้แล้ว (400 ชื่อไม่ผ่าน/ยังไม่มีป้ายพื้นที่เวกเตอร์, 404
      // รายการหลุดคิวไปแล้ว, 500 ตัดคิวไม่สำเร็จ) -- อ่านโค้ดออกมาเก็บไว้ (ไม่ใช่แค่ true)
      // เพื่อให้ pendingHtml() เลือกข้อความที่ตรงกว่า spkConfirmError ทั่วไปได้เมื่อจำเป็น
      // (เช่น missing_embedding_model ที่ "ลองใหม่" ไม่ช่วยอะไรเลย) คงชื่อที่พิมพ์ค้างไว้
      // ให้กดลองใหม่ได้ ไม่ล้างคิวทิ้ง
      const body = await response.json().catch(() => ({}));
      speakerErrors[key] = body.error || true;
      render(lastState);
      return;
    }
  } catch (e) {
    offline = true;
  }
  delete nameDrafts[key];
  delete speakerErrors[key];
  stopPlayback();
  await refreshPending();
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
  // ซ่อนตอนถอดเสียงอย่างเดียว: profile เลือกแค่ว่าจะใช้กฎสรุปชุดไหน โหมดนี้ไม่มีสรุป
  // ให้ใช้กฎกับมันเลย ถามไปก็เป็นคำถามที่คำตอบไม่เปลี่ยนอะไร (เหมือน start-meeting.bat)
  const profileOptions =
    model === NO_SUMMARY_MODEL
      ? ""
      : `<label class="field" style="margin-top:14px">${esc(x.kind)}</label>` +
        x.profiles
          .map(
            ([id, title, desc]) => `
      <div class="opt ${profile === id ? "on" : ""}" data-profile="${id}">
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
  return `${warningsHtml(state)}${workerNote}${pendingHtml()}
    <label class="field">${esc(x.mode)}</label>
    ${options}
    ${profileOptions}
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
  const micMutedNote = state.mic_muted
    ? `<div class="note warn">⚠ ${esc(x.micMutedNote)}</div>`
    : "";
  return `${warningsHtml(state)}${micMutedNote}
    <div class="rec-wrap">
      <span class="rec-badge"><span class="dot"></span>${esc(x.rec)}</span>
      <div class="timer" id="clock">${fmtClock(state.elapsed_seconds || 0)}</div>
      <div class="sub">${esc(state.room || x.untitled)} · ${esc(modelLabel || "")}</div>
      ${deviceRows}
      <button class="plain mic-toggle ${state.mic_muted ? "muted" : ""}" id="micToggle" ${isStopping ? "disabled" : ""}>
        ${esc(state.mic_muted ? x.unmuteMic : x.muteMic)}
      </button>
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
    </div>
    ${pendingHtml()}`;
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
    state.mic_muted,
    model,
    // ต้องอยู่ในลายเซ็นคู่กับ model: render() วาด body ใหม่เฉพาะเมื่อลายเซ็นเปลี่ยน
    // ถ้าไม่มีค่านี้ การกดเลือกประเภทประชุมจะเปลี่ยนตัวแปรจริงแต่เครื่องหมายถูก
    // ไม่ขยับ ผู้ใช้เห็นว่ากดไม่ติดแล้วกดซ้ำ หรือเริ่มอัดด้วยประเภทที่ไม่ได้ตั้งใจ
    profile,
    lang,
    stopping,
    progress ? `${progress.stage}:${progress.failed}` : "",
    pendingMeetings.map((m) => `${m.meeting_dir}:${m.speakers.length}`).join(","),
    Object.keys(speakerErrors).join(","),
    playingKey,
    (state.warnings || []).map((w) => w.code).join(","),
  ].join("|");
}

function render(state) {
  const x = t();
  el("hTitle").textContent = x.title;
  el("enrollLink").textContent = x.enrollLink;
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
  // ต้องเจาะจง [data-model] ไม่ใช่ .opt เฉยๆ: ตัวเลือกประเภทประชุมใช้คลาส .opt
  // เหมือนกัน ถ้าจับด้วย .opt การกดเลือกประเภทจะตั้ง model = undefined ไปด้วย
  document.querySelectorAll("[data-model]").forEach((o) => {
    o.onclick = () => {
      model = o.dataset.model;
      render(lastState);
    };
  });
  document.querySelectorAll("[data-profile]").forEach((o) => {
    o.onclick = () => {
      profile = o.dataset.profile;
      render(lastState);
    };
  });
  const room = el("room");
  if (room) room.oninput = () => (roomDraft = room.value);
  const go = el("go");
  if (go) go.onclick = openRoom;
  const stop = el("stop");
  if (stop) stop.onclick = () => el("scrim").classList.remove("hide");
  const micToggle = el("micToggle");
  if (micToggle) micToggle.onclick = () => toggleMic(!lastState.mic_muted);
  const again = el("again");
  if (again)
    again.onclick = () => {
      dismissedJob = followingJob;
      render(lastState);
    };
  document.querySelectorAll("[data-name]").forEach((input) => {
    input.oninput = () => (nameDrafts[input.dataset.name] = input.value);
  });
  document.querySelectorAll("[data-save]").forEach((button) => {
    button.onclick = () => confirmSpeaker(button.dataset.save, false);
  });
  document.querySelectorAll("[data-skip]").forEach((button) => {
    button.onclick = () => confirmSpeaker(button.dataset.skip, true);
  });
  document.querySelectorAll("[data-play]").forEach((button) => {
    button.onclick = () => {
      if (playingKey === button.dataset.play) {
        stopPlayback();
        render(lastState);
      } else {
        playSample(button.dataset.play);
      }
    };
  });
}

async function openRoom() {
  const room = el("room");
  roomDraft = room ? room.value.trim() : "";
  try {
    const response = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, profile, name: roomDraft }),
    });
    // 409 = มีห้องเปิดอยู่แล้ว ให้สถานะจริงชนะ ไม่ต้องเดา
    if (response.status === 201) {
      followingJob = null;
      dismissedJob = null;
      pendingSignalJob = null;
      roomDraft = "";
    }
  } catch (e) {
    offline = true;
  }
  await poll();
}

async function toggleMic(muted) {
  try {
    await fetch("/api/session/mic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ muted }),
    });
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
  // ดึงคิวตั้งชื่อทุก 15 วินาทีตอนไม่ได้อัด -- มันเปลี่ยนแค่ตอนจบประชุม การถามทุก
  // วินาทีคือการอ่านไฟล์ซ้ำ 60 ครั้งเพื่อคำตอบเดิม
  pendingTick += 1;
  if (state.recorder === "idle" && pendingTick % 15 === 1) refreshPending();
  // ...แต่ตอนเพิ่งปิดห้องรอ 15 วินาทีไม่ได้ ผู้ใช้กำลังมองหน้าจอ "บันทึกเรียบร้อย" อยู่
  // ตรงนั้น และคิวถูกเขียนหลัง meeting_done เล็กน้อย (ต้องรอโมเดลเดาชื่อก่อน) จึงยัง
  // ไม่มีตอนหน้าจบโผล่ครั้งแรก ดึงทันทีที่เห็นเหตุการณ์ speakers_pending ของงานนี้
  const signalled = (state.activity || []).some(
    (e) => e.code === "speakers_pending" && e.job === followingJob
  );
  if (signalled && pendingSignalJob !== followingJob) {
    pendingSignalJob = followingJob;
    refreshPending();
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
refreshPending();
setInterval(poll, 1000);
