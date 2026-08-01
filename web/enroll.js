// หน้าลงทะเบียนเสียง -- เป็นอิสระจาก app.js ทั้งหมดโดยเจตนา หน้าจอ idle ของตัวรัน
// ไม่ถูกแตะเลยแม้แต่บรรทัดเดียว สิ่งเดียวที่สองหน้าใช้ร่วมกันคือ style.css กับคีย์ภาษา
const UI = {
  th: {
    title: "ลงทะเบียนเสียงผู้พูด",
    homeLink: "กลับหน้าหลัก",
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
    rename: "แก้ชื่อ",
    renameSave: "บันทึก",
    renameCancel: "ยกเลิก",
    renamed: 'เปลี่ยนชื่อเป็น "{name}" แล้ว',
    errDuplicateName: 'มี "{name}" อยู่ในทะเบียนแล้ว ใช้ชื่อซ้ำกันไม่ได้',
    errBadName: "ชื่อนี้ใช้ไม่ได้ ลองใหม่อีกที",
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
    savedButMoveManually:
      "บันทึก {name} เข้าทะเบียนแล้ว แต่ย้ายไฟล์เข้า enroll\\done\\ ไม่สำเร็จ " +
      "กรุณาย้ายไฟล์ด้วยตัวเอง",
    errName: "ชื่อนี้ใช้ไม่ได้ ลองใหม่",
    errSave: "บันทึกไม่สำเร็จ ไฟล์ยังอยู่ที่เดิม ลองใหม่ได้",
    // ผลวิเคราะห์ที่มาจากก่อนอัปเดตระบบจำเสียง (หรือถูกแก้มือ) ไม่มีป้ายพื้นที่เวกเตอร์เลย
    // -- ต้องบอกให้วิเคราะห์ใหม่ ไม่ใช่พูดเหมือนระบบพัง (missing_embedding_model)
    errMissingEmbeddingModel:
      "ผลวิเคราะห์นี้มาจากก่อนอัปเดตระบบจำเสียง ไม่มีข้อมูลพอจะบันทึกได้ -- กดวิเคราะห์ไฟล์นี้ใหม่อีกครั้ง",
    errLoad: "โหลดรายการไฟล์ไม่สำเร็จ อาจเป็นเพราะเซิร์ฟเวอร์หยุดทำงานหรือการเชื่อมต่อขาด",
    retry: "ลองใหม่",
    errAction: "การทำงานนี้ไม่สำเร็จ อาจเป็นเพราะการเชื่อมต่อขาด ลองใหม่อีกครั้ง",
    reasonUnknown: "ไฟล์นี้ใช้ลงทะเบียนไม่ได้ด้วยเหตุผลที่ระบบยังไม่รู้จัก ลองไฟล์อื่นหรือติดต่อผู้ดูแล",
    reason_multiple_speakers:
      "ไฟล์นี้มีมากกว่าหนึ่งคน ลงทะเบียนไม่ได้เพราะระบบไม่รู้ว่าคุณหมายถึงใคร — " +
      "ตัดเอาเฉพาะช่วงที่คนเดียวพูดแล้ววางใหม่",
    reason_too_short:
      "เสียงพูดจริงสั้นเกินไป (ต่ำกว่า {min} วินาที) เวกเตอร์จากเสียงไม่กี่วินาที" +
      "เทียบข้ามการประชุมไม่ได้ — อัดใหม่ให้ยาวขึ้น แนะนำเกิน 30 วินาที",
    reason_unusable_embedding:
      "ถอดเวกเตอร์เสียงจากไฟล์นี้ไม่ได้ ลองใช้ไฟล์ที่เสียงชัดกว่านี้",
    reason_analysis_failed: "วิเคราะห์ไฟล์นี้ไม่สำเร็จ",
    // finding 5 ของรีวิวรอบนี้: ไฟล์เสียงเปลี่ยนระหว่างที่กำลังวิเคราะห์อยู่พอดี (คนละไบต์
    // กับที่ผลนี้อ้างถึง) -- ผลถูกทิ้งไปเงียบ ๆ ที่ src/enroll.py (write_result) การ์ดจึงเด้ง
    // กลับไป "รอวิเคราะห์" เอง ข้อความนี้อธิบายเหตุผลให้ผู้ใช้เห็นสักครั้งหนึ่งแทนความเงียบ
    changedDuringAnalysis:
      "ไฟล์เสียงมีการเปลี่ยนแปลงระหว่างที่กำลังวิเคราะห์อยู่ ผลก่อนหน้าจึงถูกทิ้งไป " +
      "กรุณากดวิเคราะห์อีกครั้ง",
    // ความคล้ายกับคนที่มีอยู่แล้วในทะเบียน (finding B ของรีวิวรอบสุดท้าย) -- คำนวณที่
    // เซิร์ฟเวอร์เท่านั้น หน้านี้แค่โชว์ชื่อกับคะแนนที่ปัดมาแล้ว
    matchSimilar: 'คล้ายกับ "{name}" ที่มีอยู่แล้วในทะเบียน ({score})',
    matchMerge:
      'บันทึกด้วยชื่อ "{name}" จะรวมตัวอย่างเสียงนี้เข้ากับคนคนนั้นทันที ' +
      "ไม่ได้สร้างคนใหม่",
    // แถบยืนยันก่อนลบคนออกจากทะเบียน (finding A ของรีวิวรอบสุดท้าย) -- คีย์เดียวกับ
    // รูปแบบที่ app.js ใช้ยืนยันก่อนปิดห้อง
    delTitle: 'ลบ "{name}" ออกจากทะเบียนใช่ไหม',
    delBody: "จะลบตัวอย่างเสียงที่เก็บไว้ทั้งหมด {n} ตัวอย่าง กู้คืนไม่ได้",
    delCancel: "ยกเลิก",
    delConfirm: "ลบเลย",
  },
  en: {
    title: "Enroll speaker voices",
    homeLink: "Back to main",
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
    rename: "Rename",
    renameSave: "Save",
    renameCancel: "Cancel",
    renamed: 'Renamed to "{name}"',
    errDuplicateName: '"{name}" is already in the registry — names must be unique',
    errBadName: "That name cannot be used, try another",
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
    savedButMoveManually:
      "Saved {name} to the registry, but could not move the file into " +
      "enroll\\done\\. Please move it by hand",
    errName: "That name cannot be used, try another",
    errSave: "Could not save. The file is untouched, you can try again",
    // Analysis results from before the voice-recognition upgrade (or hand-edited) carry
    // no embedding-space stamp -- say "analyze again", not "system is broken"
    errMissingEmbeddingModel:
      "This analysis predates the voice-recognition upgrade and lacks what's needed to save " +
      "-- analyze this file again",
    errLoad: "Could not load the file list. The server may be down or the connection dropped",
    retry: "Retry",
    errAction: "That did not go through, possibly a dropped connection. Try again",
    reasonUnknown: "This file cannot be enrolled for a reason this page does not recognize yet. Try another file or contact an admin",
    reason_multiple_speakers:
      "More than one person speaks in this file, so there is no way to tell who you mean — " +
      "trim it down to a stretch where only one person talks",
    reason_too_short:
      "Too little actual speech (under {min} seconds). A vector from a few seconds cannot be " +
      "compared across meetings — record a longer clip, over 30 seconds is best",
    reason_unusable_embedding:
      "No usable voice vector could be extracted. Try a cleaner recording",
    reason_analysis_failed: "Analyzing this file failed",
    changedDuringAnalysis:
      "The audio file changed while it was being analyzed, so the previous result was " +
      "discarded. Please analyze it again.",
    matchSimilar: 'Similar to "{name}", already in the registry ({score})',
    matchMerge:
      'Saving under the name "{name}" will merge this sample into that person right away, ' +
      "not create a new one",
    delTitle: 'Delete "{name}" from the registry?',
    delBody: "This removes all {n} saved voice samples. This cannot be undone.",
    delCancel: "Cancel",
    delConfirm: "Delete",
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
// ค่าเดียวกับ src/speakers.py:MIN_SPEAKING_SECONDS มาจาก /api/enroll เสมอ (finding 5:
// ห้ามฝังตัวเลขซ้ำเป็นสตริงคงที่) ค่าเริ่มต้นนี้ใช้แค่ก่อนโหลดครั้งแรกสำเร็จเท่านั้น
let minSpeakingSeconds = 10;
// notice เก็บทั้งข้อความและสถานะ (สำเร็จ/ล้มเหลว) ไว้ในก้อนเดียวกันเสมอ -- ห้ามมีตัวแปร
// severity แยกต่างหาก เพราะจุดที่ set ข้อความมีหลายที่ (save/analyze/dismiss/remove)
// ถ้าแยกกันจะมีจุดที่ set ข้อความแต่ลืม set severity ได้ง่าย ๆ
let notice = null; // { text, ok } | null
let loadError = null;
// เก็บ handle ของ poll ที่ตั้งไว้ -- ยกเลิกของเก่าก่อนตั้งใหม่เสมอ กัน chain
// ซ้อนกันหลายสายเวลา load() ถูกเรียกซ้ำจากปุ่มต่าง ๆ ระหว่างที่ยังมีคิวค้าง
let pollHandle = null;
// คนที่กด "ลบ" ค้างไว้รอยืนยัน (finding A ของรีวิวรอบสุดท้าย) -- ต้องมีขั้นยืนยันก่อน
// ลบจริง เพราะปุ่มนี้ล้างตัวอย่างเสียงสะสมได้ถึง 10 ตัวอย่างในคลิกเดียว และอยู่ติดกับ
// การ์ดไฟล์ที่ผู้ใช้ไล่กดอยู่แล้ว | { id, name, sampleCount } | null
let pendingDelete = null;
// คนที่กำลังแก้ชื่ออยู่ -- เก็บแค่ id ไม่เก็บข้อความที่พิมพ์ค้าง เพราะ render() วาด
// แถวใหม่ทุกครั้ง ค่าที่พิมพ์อยู่จึงอ่านจาก input ตอนกดบันทึกเท่านั้น | id | null
let editingId = null;

function setNotice(text, ok) {
  notice = { text, ok };
}

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
    if (typeof body.min_speaking_seconds === "number")
      minSpeakingSeconds = body.min_speaking_seconds;
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
  if (busy) {
    // ล้าง timer เดิมอีกครั้งตรงจังหวะตั้งใหม่ ไม่ใช่แค่ตอนเข้าฟังก์ชันเท่านั้น --
    // ถ้า load() สองสายแข่งกัน (เช่นผู้ใช้กดปุ่มพอดีกับที่ auto-poll ยิงครบสองวินาที)
    // ทั้งคู่ผ่านจุดล้างตอนเข้าไปได้ก่อนที่ใครจะ await fetch เสร็จ เพราะตอนนั้น
    // pollHandle ยังเป็น null อยู่ทั้งคู่ -- ถ้าล้างแค่ตอนเข้าอย่างเดียว การตั้ง
    // timer ครั้งหลังจะเขียนทับ pollHandle ทิ้ง timer ของครั้งแรกไว้แบบไม่มีใคร
    // อ้างถึงมันได้อีก (orphan) ทำให้ยัง poll ถี่เป็นสองเท่าต่อไปตลอดช่วง busy
    if (pollHandle !== null) clearTimeout(pollHandle);
    pollHandle = setTimeout(load, 2000);
  }
}

function chipFor(file) {
  if (file.state === "idle") return ["wait", t().stateIdle];
  if (file.state === "queued") return ["now", t().stateBusy];
  if (file.status === "ok") return ["ok", t().stateReady];
  return ["bad", t().stateBad];
}

function dismissButton(file) {
  // ทุกสถานะต้องมีทางออกที่ไม่ใช่การไปลบไฟล์เองใน Explorer (finding 1 ของรีวิวรอบ
  // สุดท้าย): ไฟล์ที่ลบผ่าน Explorer ทิ้ง sidecar กำพร้าไว้ ซึ่งพร้อมผูกผิดกับไฟล์ใหม่
  // ชื่อเดียวกันที่วางเข้ามาทีหลัง -- ปุ่มนี้จึงต้องโผล่ในทุกการ์ด ไม่ใช่แค่ตอนถูกปฏิเสธ
  // พฤติกรรมเดิมคงไว้ทั้งหมด: archive เข้า done/ ไม่แตะทะเบียน
  const button = node("button", "plain", t().dismiss);
  button.onclick = async () => {
    try {
      const response = await fetch(`/api/enroll/${encodeURIComponent(file.audio_file)}`, {
        method: "DELETE",
      });
      if (!response.ok) setNotice(t().errAction, false);
    } catch (err) {
      setNotice(t().errAction, false);
    }
    load();
  };
  return button;
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

  if (file.state !== "done") {
    // finding 5 ของรีวิวรอบนี้: server ล้าง sidecar ทิ้งเงียบ ๆ เมื่อผูกผลกับไฟล์เสียงไม่ได้
    // (ไฟล์เปลี่ยนระหว่างวิเคราะห์) แล้วส่งธงนี้มาครั้งเดียว (ดู enroll.list_entries /
    // _consume_changed_marker) -- ต้องโชว์ก่อนคืนการ์ด ไม่งั้นผู้ใช้เห็นแค่ "รอวิเคราะห์"
    // เฉย ๆ โดยไม่รู้ว่าทำไมงานที่ทำไปหลายนาทีถึงหายไป
    if (file.changed_during_analysis) {
      box.append(node("div", "note warn", t().changedDuringAnalysis));
    }
    box.append(dismissButton(file));
    return box;
  }

  if (file.status !== "ok") {
    const reasonKey = `reason_${file.reason}`;
    const knownReason = t()[reasonKey];
    // เหตุผลที่ไม่รู้จักต้องไม่โผล่เป็นโค้ดดิบให้ผู้ใช้เห็นตรง ๆ -- ถ้าไม่มีคำแปล
    // ให้ใช้ประโยคกลาง ๆ แทน แล้วค่อยโชว์โค้ดแยกไว้ต่างหากแบบไม่ปนกับข้อความ error
    // fill() แทน {min} เฉพาะข้อความที่มีมัน (reason_too_short) -- ข้อความอื่นไม่มี
    // placeholder นี้เลยไม่ถูกแตะ
    const reasonText = knownReason
      ? fill(knownReason, { min: minSpeakingSeconds })
      : t().reasonUnknown;
    const why = node("div", "note warn", reasonText);
    box.append(why);
    if (!knownReason) box.append(node("div", "meta", file.reason));
    box.append(dismissButton(file));
    return box;
  }

  // ความคล้ายกับคนที่มีอยู่แล้วในทะเบียน (finding B ของรีวิวรอบสุดท้าย) -- คำนวณที่
  // เซิร์ฟเวอร์แล้วเท่านั้น (/api/enroll) ที่นี่แค่โชว์ชื่อกับคะแนนที่ปัดมาให้ ไม่มี
  // เวกเตอร์อะไรมาถึงฝั่งนี้เลย ชื่อคนมาจากทะเบียนผ่าน fill()/textContent เสมอ
  if (file.match) {
    const score = file.match.score.toFixed(2);
    box.append(
      node("div", "hint", fill(t().matchSimilar, { name: file.match.name, score }))
    );
    if (file.match.confident) {
      box.append(node("div", "hint", fill(t().matchMerge, { name: file.match.name })));
    }
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
        // archive_failed = ทะเบียนบันทึกสำเร็จแล้วจริง แต่ย้ายไฟล์เข้า done/ ไม่ได้
        // (finding 2 ของรีวิวรอบสุดท้าย) -- ต้องบอกผู้ใช้ให้ย้ายเอง ไม่ใช่แกล้งโกหกว่า
        // ทุกอย่างจบสมบูรณ์เหมือนตอน savedAs ปกติ
        const text =
          body.warning === "archive_failed"
            ? fill(t().savedButMoveManually, { name: body.name })
            : fill(t().savedAs, { name: body.name });
        setNotice(text, body.warning !== "archive_failed");
      } else {
        // finding: missing_embedding_model ต้องได้ข้อความของตัวเอง ไม่ใช่ตกไปที่ errSave
        // ทั่วไปที่บอกให้ "ลองใหม่" -- กดยืนยันซ้ำไม่มีทางช่วยอะไรเลยตราบใดที่ผลนี้ยังไม่มี
        // ป้ายพื้นที่เวกเตอร์ ต้องวิเคราะห์ใหม่เท่านั้น
        const message =
          body.error === "bad_name"
            ? t().errName
            : body.error === "missing_embedding_model"
            ? t().errMissingEmbeddingModel
            : t().errSave;
        setNotice(message, false);
      }
    } catch (err) {
      // fetch เองพังก่อนถึง response ได้ (เน็ตหลุด/เซิร์ฟเวอร์ตาย) -- ต้องไม่ปล่อย
      // ปุ่มค้าง disabled อยู่แบบนั้นตลอดไปโดยไม่มีข้อความอะไรเลย
      setNotice(t().errSave, false);
    }
    // load() เรียก render() เสมอไม่ว่าจะสำเร็จหรือพัง ปุ่มใหม่จึงไม่ disabled ค้าง
    load();
  };
  row.append(input, save);
  box.append(row);
  box.append(dismissButton(file));
  return box;
}

function render() {
  el("hTitle").textContent = t().title;
  el("homeLink").textContent = t().homeLink;
  el("langBtn").textContent = t().lang;
  document.title = t().title;
  el("wText").textContent = worker ? t().workerOn : t().workerOff;
  el("wDot").classList.toggle("off", !worker);

  // แถบยืนยันก่อนลบ -- คนละคำถามในแต่ละครั้งตามคนที่กดลบ จึงต้องตั้งข้อความใหม่ทุก
  // รอบ ไม่ใช่ค่าคงที่แบบ cfTitle/cfBody ของ app.js ชื่อคนมาจากทะเบียน ผ่าน textContent
  // เสมอ ไม่ประกอบเป็น HTML
  el("cfNo").textContent = t().delCancel;
  el("cfYes").textContent = t().delConfirm;
  if (pendingDelete) {
    el("cfTitle").textContent = fill(t().delTitle, { name: pendingDelete.name });
    el("cfBody").textContent = fill(t().delBody, { n: pendingDelete.sampleCount });
    el("scrim").classList.remove("hide");
  } else {
    el("scrim").classList.add("hide");
  }

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
    // severity มากับ notice เอง (ตั้งผ่าน setNotice เท่านั้น) -- สำเร็จได้ note ok
    // (เขียว) ล้มเหลวได้ note warn (เตือน) ห้ามใช้ note ok เป็นค่าเริ่มต้นเด็ดขาด
    // เพราะข้อความ error หลายจุด (errSave/errName/errAction) มาลงที่ notice เดียวกัน
    body.append(node("div", notice.ok ? "note ok" : "note warn", notice.text));
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
        if (!response.ok) setNotice(t().errAction, false);
      } catch (err) {
        setNotice(t().errAction, false);
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
    if (editingId === speaker.id) {
      // โหมดแก้ชื่อ: แทนที่ทั้งแถวด้วยช่องกรอก ไม่ใช่แถบยืนยันแยกแบบตอนลบ -- การลบ
      // ต้องถามซ้ำเพราะกู้ไม่ได้ ส่วนการแก้ชื่อพิมพ์ผิดก็แก้ใหม่ได้ ไม่ต้องมีขั้นถาม
      const input = node("input", "renameInput");
      // node() ไม่ตั้ง type ให้ และ CSS ของหน้านี้ใช้ selector input[type=text]
      // ทั้งหมด -- ไม่ตั้งตรงนี้ ช่องกรอกจะโผล่มาแบบไม่มีสไตล์เลย
      input.type = "text";
      input.value = speaker.name;
      input.setAttribute("aria-label", t().rename);
      const save = node("button", "plain", t().renameSave);
      const cancel = node("button", "del", t().renameCancel);
      save.onclick = () => submitRename(speaker, input.value);
      cancel.onclick = () => {
        editingId = null;
        render();
      };
      input.onkeydown = (event) => {
        if (event.key === "Enter") submitRename(speaker, input.value);
        if (event.key === "Escape") {
          editingId = null;
          render();
        }
      };
      row.append(input, save, cancel);
      body.append(row);
      // วาดเสร็จแล้วค่อยโฟกัส ไม่งั้นโฟกัสไปตกที่ node ที่ยังไม่ได้อยู่ในหน้า
      setTimeout(() => {
        input.focus();
        input.select();
      }, 0);
      return;
    }
    row.append(
      node("span", null, speaker.name),
      node("span", "spacer"),
      node("span", "n", `${speaker.sample_count} ${t().samples}`)
    );
    const rename = node("button", "del", t().rename);
    rename.onclick = () => {
      editingId = speaker.id;
      // ปิดแถบยืนยันลบที่อาจค้างอยู่ ไม่งั้นผู้ใช้เห็นคำถาม "ลบใช่ไหม" ค้างอยู่
      // ข้างบนพร้อมกับช่องแก้ชื่อของอีกคน ซึ่งอ่านแล้วเข้าใจผิดได้ง่าย
      pendingDelete = null;
      render();
    };
    row.append(rename);
    const remove = node("button", "del", t().remove);
    // ไม่ยิง DELETE ทันที (finding A) -- แค่เปิดแถบยืนยัน ตัวจริงอยู่ที่ el("cfYes")
    // ข้างล่าง ซึ่งอ่าน pendingDelete ตอนกดยืนยันเท่านั้น
    remove.onclick = () => {
      pendingDelete = {
        id: speaker.id,
        name: speaker.name,
        sampleCount: speaker.sample_count,
      };
      render();
    };
    row.append(remove);
    body.append(row);
  });
}

async function submitRename(speaker, value) {
  const next = String(value || "").trim();
  // ชื่อเดิมเป๊ะ ๆ = ไม่ได้แก้อะไร ปิดโหมดแก้ไปเฉย ๆ ดีกว่ายิง request ที่ไม่ทำอะไร
  // แล้วขึ้นข้อความ "เปลี่ยนชื่อแล้ว" ทั้งที่ไม่มีอะไรเปลี่ยน
  if (!next || next === speaker.name) {
    editingId = null;
    render();
    return;
  }
  editingId = null;
  try {
    const response = await fetch(`/api/speakers/${encodeURIComponent(speaker.id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: next }),
    });
    if (response.ok) {
      const body = await response.json();
      // ใช้ชื่อที่เซิร์ฟเวอร์คืนมา ไม่ใช่ที่ผู้ใช้พิมพ์ -- clean_name ฝั่งนั้นตัด
      // อักขระที่ทำให้ markdown เสียรูปออก ผู้ใช้ต้องเห็นชื่อที่ถูกบันทึกจริง
      setNotice(fill(t().renamed, { name: (body.speaker || {}).name || next }), true);
    } else if (response.status === 409) {
      setNotice(fill(t().errDuplicateName, { name: next }), false);
    } else if (response.status === 400) {
      setNotice(t().errBadName, false);
    } else {
      setNotice(t().errAction, false);
    }
  } catch (err) {
    setNotice(t().errAction, false);
  }
  load();
}

el("langBtn").onclick = () => {
  lang = lang === "th" ? "en" : "th";
  // คีย์เดียวกับ app.js -- สลับภาษาที่หน้าไหนแล้วอีกหน้าจำตาม
  localStorage.setItem("runnerLang", lang);
  render();
};

// รูปแบบเดียวกับ el("cfNo")/el("cfYes") ใน app.js: ผูกครั้งเดียวตรงนี้ อ่านสถานะ
// ปัจจุบัน (pendingDelete) ตอนถูกกด ไม่ใช่ตอนวาด
el("cfNo").onclick = () => {
  pendingDelete = null;
  render();
};
el("cfYes").onclick = async () => {
  const target = pendingDelete;
  pendingDelete = null;
  if (!target) {
    render();
    return;
  }
  try {
    const response = await fetch(`/api/speakers/${encodeURIComponent(target.id)}`, {
      method: "DELETE",
    });
    if (!response.ok) setNotice(t().errAction, false);
  } catch (err) {
    setNotice(t().errAction, false);
  }
  load();
};

load();
