import {
  PoseLandmarker,
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// --- tunable thresholds ---
const NECK_ANGLE_THRESHOLD_DEG = 12; // deviation from calibrated baseline
const BACK_ANGLE_THRESHOLD_DEG = 10;
const WAIST_ANGLE_THRESHOLD_DEG = 10;
const SHOULDER_ASYMMETRY_THRESHOLD = 0.06; // ratio of shoulder width
const SUSTAIN_MS = 3000; // deviation must persist this long before alerting
const SMOOTHING_WINDOW = 8; // frames averaged to reduce jitter

const LM = {
  NOSE: 0,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
};

// wrist/finger points from the pose model are too coarse (only 3 points per hand) -
// the dedicated HandLandmarker draws real 5-finger hands instead, so skip these here
const HAND_POINT_INDICES = new Set([17, 18, 19, 20, 21, 22]);
const BODY_CONNECTIONS = PoseLandmarker.POSE_CONNECTIONS.filter(
  (c) => !HAND_POINT_INDICES.has(c.start) && !HAND_POINT_INDICES.has(c.end)
);

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const calibrateBtn = document.getElementById("calibrateBtn");
const calibrateStatus = document.getElementById("calibrateStatus");
const alertBanner = document.getElementById("alertBanner");
const neckStatusEl = document.getElementById("neckStatus");
const row2Label = document.getElementById("row2Label");
const row2StatusEl = document.getElementById("row2Status");
const shoulderRow = document.getElementById("shoulderRow");
const shoulderStatusEl = document.getElementById("shoulderStatus");
const modeFrontBtn = document.getElementById("modeFrontBtn");
const modeSideBtn = document.getElementById("modeSideBtn");
const detectionIndicator = document.getElementById("detectionIndicator");
const detectionText = document.getElementById("detectionText");

let poseLandmarker = null;
let handLandmarker = null;
let mode = "front"; // "front" | "side"
let baselines = { front: null, side: null };
let metricBuffer = [];
let badSince = { neck: null, back: null, shoulder: null, waist: null };
let calibrating = false;
let calibrationSamples = [];

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// angle of the p1->p2 vector from straight-up vertical, in degrees
function angleFromVertical(p1, p2) {
  const dx = p2.x - p1.x;
  const dy = p1.y - p2.y; // image y grows downward, so "up" is negative dy
  return Math.atan2(dx, dy) * (180 / Math.PI);
}

// unsigned angle (deg) at vertex `v` between rays to `a` and `b`
function angleAtVertex(v, a, b) {
  const v1 = { x: a.x - v.x, y: a.y - v.y };
  const v2 = { x: b.x - v.x, y: b.y - v.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag = Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y) || 1;
  const cos = Math.min(1, Math.max(-1, dot / mag));
  return Math.acos(cos) * (180 / Math.PI);
}

function toPixelSpace(landmarks) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  return landmarks.map((p) => ({ x: p.x * w, y: p.y * h, visibility: p.visibility }));
}

function pickVisibleSide(px) {
  const leftVis =
    (px[LM.LEFT_EAR].visibility + px[LM.LEFT_SHOULDER].visibility + px[LM.LEFT_HIP].visibility + px[LM.LEFT_KNEE].visibility) / 4;
  const rightVis =
    (px[LM.RIGHT_EAR].visibility + px[LM.RIGHT_SHOULDER].visibility + px[LM.RIGHT_HIP].visibility + px[LM.RIGHT_KNEE].visibility) / 4;
  return leftVis >= rightVis ? "left" : "right";
}

function computeMetricsFront(px) {
  const leftEar = px[LM.LEFT_EAR];
  const rightEar = px[LM.RIGHT_EAR];
  const leftShoulder = px[LM.LEFT_SHOULDER];
  const rightShoulder = px[LM.RIGHT_SHOULDER];
  const leftHip = px[LM.LEFT_HIP];
  const rightHip = px[LM.RIGHT_HIP];

  const earMid = midpoint(leftEar, rightEar);
  const shoulderMid = midpoint(leftShoulder, rightShoulder);
  const hipMid = midpoint(leftHip, rightHip);
  const shoulderWidth = distance(leftShoulder, rightShoulder) || 1;

  return {
    neckAngle: angleFromVertical(shoulderMid, earMid),
    backAngle: angleFromVertical(hipMid, shoulderMid),
    shoulderAsymmetry: Math.abs(leftShoulder.y - rightShoulder.y) / shoulderWidth,
  };
}

// side-view (profile) metrics: uses whichever side of the body is more visible to the camera
function computeMetricsSide(px) {
  const side = pickVisibleSide(px);
  const ear = side === "left" ? px[LM.LEFT_EAR] : px[LM.RIGHT_EAR];
  const shoulder = side === "left" ? px[LM.LEFT_SHOULDER] : px[LM.RIGHT_SHOULDER];
  const hip = side === "left" ? px[LM.LEFT_HIP] : px[LM.RIGHT_HIP];
  const knee = side === "left" ? px[LM.LEFT_KNEE] : px[LM.RIGHT_KNEE];

  return {
    neckAngle: angleFromVertical(shoulder, ear),
    // angle at the hip between torso (hip->shoulder) and thigh (hip->knee); leaning
    // forward at the waist closes this angle
    waistAngle: angleAtVertex(hip, shoulder, knee),
  };
}

// head/neck/mid-back/hip control points for the neck+spine curve overlay
function computeSpinePoints(px) {
  let head, neck, hip;
  if (mode === "front") {
    head = midpoint(px[LM.LEFT_EAR], px[LM.RIGHT_EAR]);
    neck = midpoint(px[LM.LEFT_SHOULDER], px[LM.RIGHT_SHOULDER]);
    hip = midpoint(px[LM.LEFT_HIP], px[LM.RIGHT_HIP]);
  } else {
    const side = pickVisibleSide(px);
    head = side === "left" ? px[LM.LEFT_EAR] : px[LM.RIGHT_EAR];
    neck = side === "left" ? px[LM.LEFT_SHOULDER] : px[LM.RIGHT_SHOULDER];
    hip = side === "left" ? px[LM.LEFT_HIP] : px[LM.RIGHT_HIP];
  }
  const midBack = { x: neck.x + 0.65 * (hip.x - neck.x), y: neck.y + 0.65 * (hip.y - neck.y) };
  return { head, neck, midBack, hip };
}

function average(metrics) {
  const n = metrics.length;
  const keys = Object.keys(metrics[0]);
  const result = {};
  for (const key of keys) {
    result[key] = metrics.reduce((s, m) => s + m[key], 0) / n;
  }
  return result;
}

function setBadge(el, state, goodText, badText) {
  el.classList.remove("idle", "good", "bad");
  if (state === null) {
    el.classList.add("idle");
    el.textContent = "대기";
  } else if (state) {
    el.classList.add("bad");
    el.textContent = badText;
  } else {
    el.classList.add("good");
    el.textContent = goodText;
  }
}

function evaluateSustained(key, isDeviating, now) {
  if (isDeviating) {
    if (badSince[key] === null) badSince[key] = now;
    return now - badSince[key] >= SUSTAIN_MS;
  }
  badSince[key] = null;
  return false;
}

function updateStatusFront(smoothed, now) {
  const baseline = baselines.front;
  const neckDeviates = Math.abs(smoothed.neckAngle - baseline.neckAngle) > NECK_ANGLE_THRESHOLD_DEG;
  const backDeviates = Math.abs(smoothed.backAngle - baseline.backAngle) > BACK_ANGLE_THRESHOLD_DEG;
  const shoulderDeviates = Math.abs(smoothed.shoulderAsymmetry - baseline.shoulderAsymmetry) > SHOULDER_ASYMMETRY_THRESHOLD;

  const neckBad = evaluateSustained("neck", neckDeviates, now);
  const backBad = evaluateSustained("back", backDeviates, now);
  const shoulderBad = evaluateSustained("shoulder", shoulderDeviates, now);
  badSince.waist = null;

  setBadge(neckStatusEl, neckBad, "정상", "거북목 감지됨");
  setBadge(row2StatusEl, backBad, "정상", "등이 굽었어요");
  setBadge(shoulderStatusEl, shoulderBad, "정상", "어깨가 기울었어요");

  const issues = [];
  if (neckBad) issues.push("거북목 자세");
  if (backBad) issues.push("등이 굽음");
  if (shoulderBad) issues.push("어깨 비대칭 (척추 정렬 확인)");
  return issues;
}

function updateStatusSide(smoothed, now) {
  const baseline = baselines.side;
  const neckDeviates = Math.abs(smoothed.neckAngle - baseline.neckAngle) > NECK_ANGLE_THRESHOLD_DEG;
  const waistDeviates = Math.abs(smoothed.waistAngle - baseline.waistAngle) > WAIST_ANGLE_THRESHOLD_DEG;

  const neckBad = evaluateSustained("neck", neckDeviates, now);
  const waistBad = evaluateSustained("waist", waistDeviates, now);
  badSince.back = null;
  badSince.shoulder = null;

  setBadge(neckStatusEl, neckBad, "정상", "거북목 감지됨");
  setBadge(row2StatusEl, waistBad, "정상", "허리가 굽었어요");

  const issues = [];
  if (neckBad) issues.push("거북목 자세");
  if (waistBad) issues.push("허리 굽음");
  return issues;
}

// returns true if any posture issue is currently flagged
function updateStatus(smoothed, now) {
  if (!baselines[mode]) return false;

  const issues = mode === "front" ? updateStatusFront(smoothed, now) : updateStatusSide(smoothed, now);

  if (issues.length > 0) {
    alertBanner.textContent = `자세 교정 필요: ${issues.join(", ")}`;
    alertBanner.classList.remove("hidden");
  } else {
    alertBanner.classList.add("hidden");
  }
  return issues.length > 0;
}

function setMode(newMode) {
  mode = newMode;
  modeFrontBtn.classList.toggle("active", mode === "front");
  modeSideBtn.classList.toggle("active", mode === "side");
  row2Label.textContent = mode === "front" ? "등(굽음)" : "허리(굽음)";
  shoulderRow.classList.toggle("hidden-row", mode === "side");

  metricBuffer = [];
  badSince = { neck: null, back: null, shoulder: null, waist: null };
  setBadge(neckStatusEl, null);
  setBadge(row2StatusEl, null);
  setBadge(shoulderStatusEl, null);
  alertBanner.classList.add("hidden");

  calibrateStatus.textContent = baselines[mode]
    ? "이 모드의 기준 자세가 이미 저장되어 있습니다. 다시 저장하려면 버튼을 누르세요."
    : "바르게 앉은 후 버튼을 눌러 기준 자세를 저장하세요.";
}

function setDetectionState(active) {
  detectionIndicator.classList.toggle("active", active);
  detectionIndicator.classList.toggle("idle", !active);
  detectionText.textContent = active ? "자세 감지 중" : "사람이 보이지 않습니다";
}

function drawSpineCurve(spine, hasIssue) {
  const { head, neck, midBack, hip } = spine;
  ctx.beginPath();
  ctx.moveTo(head.x, head.y);
  ctx.bezierCurveTo(neck.x, neck.y, midBack.x, midBack.y, hip.x, hip.y);
  ctx.strokeStyle = hasIssue ? "#ff4d4d" : "#00e5ff";
  ctx.lineWidth = 4;
  ctx.lineCap = "round";
  ctx.stroke();

  for (const p of [head, neck, midBack, hip]) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = hasIssue ? "#ff4d4d" : "#00e5ff";
    ctx.fill();
  }
}

function drawSkeleton(poseLandmarksList, handResult, spine, hasIssue) {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const drawingUtils = new DrawingUtils(ctx);

  for (const landmarks of poseLandmarksList) {
    drawingUtils.drawConnectors(landmarks, BODY_CONNECTIONS, { color: "#3d7cff", lineWidth: 2 });
    const bodyDots = landmarks.filter((_, idx) => !HAND_POINT_INDICES.has(idx));
    drawingUtils.drawLandmarks(bodyDots, { color: "#ffffff", radius: 2 });
  }

  if (handResult) {
    for (const landmarks of handResult.landmarks) {
      drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#ffd23d", lineWidth: 2 });
      drawingUtils.drawLandmarks(landmarks, { color: "#ffffff", radius: 2 });
    }
  }

  if (spine) drawSpineCurve(spine, hasIssue);

  ctx.restore();
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

async function setupLandmarkers() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

function renderLoop() {
  requestAnimationFrame(renderLoop);
  if (!poseLandmarker || !handLandmarker || video.readyState < 2) return;

  const now = performance.now();
  const poseResult = poseLandmarker.detectForVideo(video, now);
  const handResult = handLandmarker.detectForVideo(video, now);

  const personVisible = poseResult.landmarks.length > 0;
  setDetectionState(personVisible);

  if (!personVisible) {
    drawSkeleton(poseResult.landmarks, handResult, null, false);
    return;
  }

  const px = toPixelSpace(poseResult.landmarks[0]);
  const spine = computeSpinePoints(px);
  const metrics = mode === "front" ? computeMetricsFront(px) : computeMetricsSide(px);

  if (calibrating) {
    calibrationSamples.push(metrics);
    drawSkeleton(poseResult.landmarks, handResult, spine, false);
    return;
  }

  metricBuffer.push(metrics);
  if (metricBuffer.length > SMOOTHING_WINDOW) metricBuffer.shift();
  const smoothed = average(metricBuffer);
  const hasIssue = updateStatus(smoothed, now);

  drawSkeleton(poseResult.landmarks, handResult, spine, hasIssue);
}

async function runCalibration() {
  calibrating = true;
  calibrationSamples = [];
  calibrateBtn.disabled = true;
  calibrateStatus.textContent = "측정 중... 자세를 유지해주세요 (2초)";

  await new Promise((resolve) => setTimeout(resolve, 2000));

  calibrating = false;
  if (calibrationSamples.length === 0) {
    calibrateStatus.textContent = "측정 실패: 카메라에 상체가 잘 보이도록 조정 후 다시 시도하세요.";
    calibrateBtn.disabled = false;
    return;
  }

  baselines[mode] = average(calibrationSamples);
  badSince = { neck: null, back: null, shoulder: null, waist: null };
  metricBuffer = [];
  calibrateStatus.textContent = "기준 자세 저장 완료! 이제부터 자세를 모니터링합니다.";
  calibrateBtn.disabled = false;
  calibrateBtn.textContent = "기준 자세 다시 저장";
}

async function main() {
  await setupCamera();
  await setupLandmarkers();
  calibrateBtn.disabled = false;
  calibrateStatus.textContent = "바르게 앉은 후 버튼을 눌러 기준 자세를 저장하세요.";
  calibrateBtn.addEventListener("click", runCalibration);
  modeFrontBtn.addEventListener("click", () => setMode("front"));
  modeSideBtn.addEventListener("click", () => setMode("side"));
  renderLoop();
}

main().catch((err) => {
  console.error(err);
  calibrateStatus.textContent = `오류: ${err.message}`;
});
