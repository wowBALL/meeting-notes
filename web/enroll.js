// หน้าลงทะเบียนเสียง -- เป็นอิสระจาก app.js ทั้งหมดโดยเจตนา หน้าจอ idle ของตัวรัน
// ไม่ถูกแตะเลยแม้แต่บรรทัดเดียว สิ่งเดียวที่สองหน้าใช้ร่วมกันคือ style.css กับคีย์ภาษา
const UI = {
  th: {
    title: "ลงทะเบียนเสียงผู้พูด",
    lang: "EN",
    workerOn: "ตัวประมวลผลพร้อม",
    workerOff: "ตัวประมวลผลไม่ได้รัน",
    dropTitle: "วางไฟล์เสียงไว้ที่ enroll\\",
    dropHint:
      "ตั้งชื่อไฟล์เป็นชื่อคน เช่น สมชาย.ogg ระบบจะเติมชื่อให้ล่วงหน้า · " +
      "รองรับ .wav .ogg .mp3 .m4a · ควรยาวเกิน 30 วินาที และมีคนพูดคนเดียว",
    refresh: "รีเฟรชรายการไฟล์",
    registryTitle: "อยู่ในทะเบียนแล้ว",
    registryEmpty: "ยังไม่มีใครในทะเบียน",
    samples: "ตัวอย่าง",
    remove: "ลบ",
    stateIdle: "รอวิเคราะห์",
    stateBusy: "กำลังวิเคราะห์",
    stateReady: "พร้อมบันทึก",
    stateBad: "ใช้ไม่ได้",
    analyzeOne: "วิเคราะห์เสียง 1 ไฟล์",
    analyzeMany: "วิเคราะห์เสียง {n} ไฟล์",
    analyzing: "กำลังวิเคราะห์…",
    queueNote:
      "งานจะถูกส่งให้ตัวประมวลผล ถ้ากำลังถอดเทปประชุมอยู่ คิวนี้จะรอจนเสร็จก่อน",
    save: "บันทึก",
    dismiss: "เอาออกจากรายการ",
    spoke: "พูดจริง {s} วินาที",
    found: "พบผู้พูด {n} คน",
    savedAs: "บันทึก {name} เข้าทะเบียนแล้ว",
    errName: "ชื่อนี้ใช้ไม่ได้ ลองใหม่",
    errSave: "บันทึกไม่สำเร็จ ไฟล์ยังอยู่ที่เดิม ลองใหม่ได้",
    errLoad: "โหลดรายการไฟล์ไม่สำเร็จ อาจเป็นเพราะเซิร์ฟเวอร์หยุดทำงานหรือการเชื่อมต่อขาด",
    retry: "ลองใหม่",
    errAction: "การทำงานนี้ไม่สำเร็จ อาจเป็นเพราะการเชื่อมต่อขาด ลองใหม่อีกครั้ง",
    reasonUnknown: "ไฟล์นี้ใช้ลงทะเบียนไม่ได้ด้วยเหตุผลที่ระบบยังไม่รู้จัก ลองไฟล์อื่นหรือติดต่อผู้ดูแล",
    reason_multiple_speakers:
      "ไฟล์นี้มีมากกว่าหนึ่งคน ลงทะเบียนไม่ได้เพราะระบบไม่รู้ว่าคุณหมายถึงใคร — " +
      "ตัดเอาเฉพาะช่วงที่คนเดียวพูดแล้ววางใหม่",
    reason_too_short:
      "เสียงพูดจริงสั้นเกินไป (ต่ำกว่า 10 วินาที) เวกเตอร์จากเสียงไม่กี่วินาที" +
      "เทียบข้ามการประชุมไม่ได้ — อัดใหม่ให้ยาวขึ้น แนะนำเกิน 30 วินาที",
    reason_unusable_embedding:
      "ถอดเวกเตอร์เสียงจากไฟล์นี้ไม่ได้ ลองใช้ไฟล์ที่เสียงชัดกว่านี้",
    reason_analysis_failed: "วิเคราะห์ไฟล์นี้ไม่สำเร็จ",
  },
  en: {
    title: "Enroll speaker voices",
    lang: "ไทย",
    workerOn: "Worker ready",
    workerOff: "Worker not running",
    dropTitle: "Drop audio files into enroll\\",
    dropHint:
      "Name the file after the person, e.g. alice.ogg — the name is filled in for you · " +
      "Accepts .wav .ogg .mp3 .m4a · Aim for over 30 seconds of one person speaking",
    refresh: "Refresh file list",
    registryTitle: "Already enrolled",
    registryEmpty: "Nobody enrolled yet",
    samples: "samples",
    remove: "Remove",
    stateIdle: "Not analyzed",
    stateBusy: "Analyzing",
    stateReady: "Ready to save",
    stateBad: "Unusable",
    analyzeOne: "Analyze 1 file",
    analyzeMany: "Analyze {n} files",
    analyzing: "Analyzing…",
    queueNote:
      "The job goes to the worker. If it is transcribing a meeting, this waits until that finishes.",
    save: "Save",
    dismiss: "Remove from list",
    spoke: "{s} seconds of speech",
    found: "{n} speaker(s) found",
    savedAs: "Saved {name} to the registry",
    errName: "That name cannot be used, try another",
    errSave: "Could not save. The file is untouched, you can try again",
    errLoad: "Could not load the file list. The server may be down or the connection dropped",
    retry: "Retry",
    errAction: "That did not go through, possibly a dropped connection. Try again",
    reasonUnknown: "This file cannot be enrolled for a reason this page does not recognize yet. Try another file or contact an admin",
    reason_multiple_speakers:
      "More than one person speaks in this file, so there is no way to tell who you mean — " +
      "trim it down to a stretch where only one person talks",
    reason_too_short:
      "Too little actual speech (under 10 seconds). A vector from a few seconds cannot be " +
      "compared across meetings — record a longer clip, over 30 seconds is best",
    reason_unusable_embedding:
      "No usable voice vector could be extracted. Try a cleaner recording",
    reason_analysis_failed: "Analyzing this file failed",
  },
};

let lang = localStorage.getItem("runnerLang") === "en" ? "en" : "th";
const t = () => UI[lang];
const el = (id) => document.getElementById(id);
const fill = (text, values) =>
  Object.entries(values).reduce((s, [k, v]) => s.split(`{${k}}`).join(v), text);

let files = [];
let speakers = [];
let worker = false;
let busy = false;
let notice = null;
let loadError = null;
// เก็บ handle ของ poll ที่ตั้งไว้ -- ยกเลิกของเก่าก่อนตั้งใหม่เสมอ กัน chain
// ซ้อนกันหลายสายเวลา load() ถูกเรียกซ้ำจากปุ่มต่าง ๆ ระหว่างที่ยังมีคิวค้าง
let pollHandle = null;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  // textContent เสมอ ไม่ใช่ innerHTML -- ชื่อไฟล์มาจากดิสก์และมีอักขระอะไรก็ได้
  if (text !== undefined) element.textContent = text;
  return element;
}

async function load() {
  // ยกเลิก poll ที่ตั้งไว้ก่อนหน้าทุกครั้งที่ load() ถูกเรียก ไม่ว่าจะเรียกจาก
  // timer เองหรือจากปุ่ม refresh/save/dismiss/remove -- กันไม่ให้เกิด chain
  // การ poll ซ้อนกันหลายสายพร้อมกัน
  if (pollHandle !== null) {
    clearTimeout(pollHandle);
    pollHandle = null;
  }
  try {
    const response = await fetch("/api/enroll");
    if (!response.ok) throw new Error(`http ${response.status}`);
    const body = await response.json();
    files = body.files || [];
    speakers = body.speakers || [];
    worker = body.worker === true;
    busy = files.some((file) => file.state === "queued");
    loadError = null;
  } catch (err) {
    // โหลดครั้งแรกล้มเหลวก็ต้องยัง render ได้ -- ไม่งั้นหน้าจะค้างเป็น
    // skeleton เปล่า ๆ ไม่มีอะไรให้กดเลย
    loadError = t().errLoad;
    busy = false;
  }
  render();
  // poll ต่อเฉพาะตอนที่มีงานค้างจริง หน้าที่นิ่งแล้วไม่ควรยิงทุกสองวินาทีตลอดไป
  if (busy) pollHandle = setTimeout(load, 2000);
}

function chipFor(file) {
  if (file.state === "idle") return ["wait", t().stateIdle];
  if (file.state === "queued") return ["now", t().stateBusy];
  if (file.status === "ok") return ["ok", t().stateReady];
  return ["bad", t().stateBad];
}

function renderFile(file) {
  const box = node("div", "file");
  const top = node("div", "top");
  top.append(node("span", "fn", file.audio_file), node("span", "spacer"));
  const [chipClass, chipText] = chipFor(file);
  top.append(node("span", `chip ${chipClass}`, chipText));
  box.append(top);

  const bits = [];
  if (file.speaking_seconds !== undefined)
    bits.push(fill(t().spoke, { s: file.speaking_seconds }));
  if (file.speaker_count !== undefined)
    bits.push(fill(t().found, { n: file.speaker_count }));
  if (!bits.length) bits.push(`${Math.round(file.size_bytes / 1024)} KB`);
  box.append(node("div", "meta", bits.join(" · ")));

  if (file.state !== "done") return box;

  if (file.status !== "ok") {
    const reasonKey = `reason_${file.reason}`;
    const knownReason = t()[reasonKey];
    // เหตุผลที่ไม่รู้จักต้องไม่โผล่เป็นโค้ดดิบให้ผู้ใช้เห็นตรง ๆ -- ถ้าไม่มีคำแปล
    // ให้ใช้ประโยคกลาง ๆ แทน แล้วค่อยโชว์โค้ดแยกไว้ต่างหากแบบไม่ปนกับข้อความ error
    const why = node("div", "note warn", knownReason || t().reasonUnknown);
    box.append(why);
    if (!knownReason) box.append(node("div", "meta", file.reason));
    const dismiss = node("button", "plain", t().dismiss);
    dismiss.onclick = async () => {
      try {
        const response = await fetch(`/api/enroll/${encodeURIComponent(file.audio_file)}`, {
          method: "DELETE",
        });
        if (!response.ok) notice = t().errAction;
      } catch (err) {
        notice = t().errAction;
      }
      load();
    };
    box.append(dismiss);
    return box;
  }

  const row = node("div", "row");
  const input = document.createElement("input");
  input.type = "text";
  input.value = file.suggested_name || "";
  const save = node("button", "primary", t().save);
  save.style.width = "auto";
  save.onclick = async () => {
    save.disabled = true;
    try {
      const response = await fetch("/api/enroll/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio_file: file.audio_file, name: input.value }),
      });
      const body = await response.json().catch(() => ({}));
      if (response.ok) {
        notice = fill(t().savedAs, { name: body.name });
      } else {
        notice = body.error === "bad_name" ? t().errName : t().errSave;
      }
    } catch (err) {
      // fetch เองพังก่อนถึง response ได้ (เน็ตหลุด/เซิร์ฟเวอร์ตาย) -- ต้องไม่ปล่อย
      // ปุ่มค้าง disabled อยู่แบบนั้นตลอดไปโดยไม่มีข้อความอะไรเลย
      notice = t().errSave;
    }
    // load() เรียก render() เสมอไม่ว่าจะสำเร็จหรือพัง ปุ่มใหม่จึงไม่ disabled ค้าง
    load();
  };
  row.append(input, save);
  box.append(row);
  return box;
}

function render() {
  el("hTitle").textContent = t().title;
  el("langBtn").textContent = t().lang;
  document.title = t().title;
  el("wText").textContent = worker ? t().workerOn : t().workerOff;
  el("wDot").classList.toggle("off", !worker);

  const body = el("body");
  body.replaceChildren();

  if (loadError) {
    // แสดง error เด่นสุดบนหัวหน้าเสมอ พร้อมปุ่มลองใหม่ที่ใช้งานได้จริง -- โหลด
    // ครั้งแรกพังไม่ควรทำให้หน้าเหลือแค่ skeleton เปล่าที่กดอะไรไม่ได้เลย
    body.append(node("div", "note warn", loadError));
    const retry = node("button", "plain", t().retry);
    retry.onclick = () => load();
    body.append(retry);
  }

  if (notice) {
    body.append(node("div", "note ok", notice));
    notice = null;
  }

  const drop = node("div", "drop");
  drop.append(node("div", null, t().dropTitle), node("div", "why", t().dropHint));
  body.append(drop);

  files.forEach((file) => body.append(renderFile(file)));

  const ready = files.filter((file) => file.state === "idle");
  if (ready.length) {
    const label =
      ready.length === 1 ? t().analyzeOne : fill(t().analyzeMany, { n: ready.length });
    const button = node("button", "primary", busy ? t().analyzing : label);
    button.disabled = busy;
    button.onclick = async () => {
      button.disabled = true;
      try {
        const response = await fetch("/api/enroll/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ files: ready.map((file) => file.audio_file) }),
        });
        if (!response.ok) notice = t().errAction;
      } catch (err) {
        notice = t().errAction;
      }
      load();
    };
    body.append(button, node("div", "note", t().queueNote));
  } else {
    // Element.append() คืน undefined -- ต้องผูก onclick กับตัว element ก่อน append
    const refresh = node("button", "plain", t().refresh);
    refresh.onclick = load;
    body.append(refresh);
  }

  body.append(
    node("div", "sec", `${t().registryTitle} · ${speakers.length}`)
  );
  if (!speakers.length) body.append(node("div", "meta", t().registryEmpty));
  speakers.forEach((speaker) => {
    const row = node("div", "reg");
    row.append(
      node("span", null, speaker.name),
      node("span", "spacer"),
      node("span", "n", `${speaker.sample_count} ${t().samples}`)
    );
    const remove = node("button", "del", t().remove);
    remove.onclick = async () => {
      try {
        const response = await fetch(`/api/speakers/${speaker.id}`, { method: "DELETE" });
        if (!response.ok) notice = t().errAction;
      } catch (err) {
        notice = t().errAction;
      }
      load();
    };
    row.append(remove);
    body.append(row);
  });
}

el("langBtn").onclick = () => {
  lang = lang === "th" ? "en" : "th";
  // คีย์เดียวกับ app.js -- สลับภาษาที่หน้าไหนแล้วอีกหน้าจำตาม
  localStorage.setItem("runnerLang", lang);
  render();
};

load();
