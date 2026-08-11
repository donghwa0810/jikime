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
const SUSTAIN_MS = 1000; // deviation must persist this long before alerting
const SMOOTHING_WINDOW = 5; // frames averaged to reduce jitter
const NECK_BEND_GAIN = 7; // visually exaggerates neck deviation from the calibrated baseline
const SPINE_BEND_GAIN = 7; // visually exaggerates spine deviation from the calibrated baseline
const SPINE_SMOOTHING_ALPHA = 0.25; // lower = smoother/slower spine bow, higher = snappier
const NO_ISSUE = { any: false, neckBad: false, spineBad: false };

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
// extra eye-corner points (1,3,4,6) clutter the face - keep only one dot per eye (2,5)
const FACE_EXTRA_INDICES = new Set([1, 3, 4, 6]);
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
// side mode picks left/right by per-frame visibility, which can flicker frame to frame and
// make the spine/neck curve look frozen/inconsistent - lock it once calibration decides it
let lockedSide = null;
let sideVotes = [];
let smoothedSpineDeviation = null;

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

// use the side locked in during calibration so the curve always tracks the same side;
// falls back to per-frame detection only before a side-mode baseline has been saved
function resolveSide(px) {
  return lockedSide || pickVisibleSide(px);
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
  const torsoLength = distance(shoulderMid, hipMid) || 1;

  return {
    neckAngle: angleFromVertical(shoulderMid, earMid),
    backAngle: angleFromVertical(hipMid, shoulderMid),
    shoulderAsymmetry: Math.abs(leftShoulder.y - rightShoulder.y) / shoulderWidth,
    // how far forward the head/shoulder sit, as a ratio of torso length (scale-invariant)
    neckForwardRatio: (earMid.x - shoulderMid.x) / torsoLength,
    spineForwardRatio: (shoulderMid.x - hipMid.x) / torsoLength,
  };
}

// side-view (profile) metrics: uses whichever side of the body is more visible to the camera
function computeMetricsSide(px) {
  const side = resolveSide(px);
  const ear = side === "left" ? px[LM.LEFT_EAR] : px[LM.RIGHT_EAR];
  const shoulder = side === "left" ? px[LM.LEFT_SHOULDER] : px[LM.RIGHT_SHOULDER];
  const hip = side === "left" ? px[LM.LEFT_HIP] : px[LM.RIGHT_HIP];
  const torsoLength = distance(shoulder, hip) || 1;

  return {
    neckAngle: angleFromVertical(shoulder, ear),
    // same style as neckAngle, applied to hip->shoulder instead of shoulder->ear - keeps
    // the waist reading independent of the knee, which is often out of frame at a desk
    waistAngle: angleFromVertical(hip, shoulder),
    // how far forward the head/shoulder sit, as a ratio of torso length (scale-invariant)
    neckForwardRatio: (ear.x - shoulder.x) / torsoLength,
    spineForwardRatio: (shoulder.x - hip.x) / torsoLength,
  };
}

// head/neck/hip control points for the separate neck-curve and spine-curve overlays
function computeSpinePoints(px) {
  if (mode === "front") {
    return {
      head: midpoint(px[LM.LEFT_EAR], px[LM.RIGHT_EAR]),
      neck: midpoint(px[LM.LEFT_SHOULDER], px[LM.RIGHT_SHOULDER]),
      hip: midpoint(px[LM.LEFT_HIP], px[LM.RIGHT_HIP]),
    };
  }
  const side = resolveSide(px);
  return {
    head: side === "left" ? px[LM.LEFT_EAR] : px[LM.RIGHT_EAR],
    neck: side === "left" ? px[LM.LEFT_SHOULDER] : px[LM.RIGHT_SHOULDER],
    hip: side === "left" ? px[LM.LEFT_HIP] : px[LM.RIGHT_HIP],
  };
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
  return { issues, neckBad, spineBad: backBad };
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
  return { issues, neckBad, spineBad: waistBad };
}

// returns { any, neckBad, spineBad } so the neck/spine curves can be colored independently
function updateStatus(smoothed, now) {
  if (!baselines[mode]) return NO_ISSUE;

  const result = mode === "front" ? updateStatusFront(smoothed, now) : updateStatusSide(smoothed, now);

  if (result.issues.length > 0) {
    alertBanner.textContent = `자세 교정 필요: ${result.issues.join(", ")}`;
    alertBanner.classList.remove("hidden");
  } else {
    alertBanner.classList.add("hidden");
  }
  return { any: result.issues.length > 0, neckBad: result.neckBad, spineBad: result.spineBad };
}

function setMode(newMode) {
  mode = newMode;
  modeFrontBtn.classList.toggle("active", mode === "front");
  modeSideBtn.classList.toggle("active", mode === "side");
  row2Label.textContent = mode === "front" ? "등(굽음)" : "허리(굽음)";
  shoulderRow.classList.toggle("hidden-row", mode === "side");

  metricBuffer = [];
  badSince = { neck: null, back: null, shoulder: null, waist: null };
  lockedSide = null;
  sideVotes = [];
  smoothedSpineDeviation = null;
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

function drawCurveSegment(from, control, to, isBad) {
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.quadraticCurveTo(control.x, control.y, to.x, to.y);
  ctx.strokeStyle = isBad ? "#ff4d4d" : "#00e5ff";
  ctx.lineWidth = 4;
  ctx.lineCap = "round";
  ctx.stroke();
}

function drawSpineCurve(spine, status, baseline) {
  const { head, neck, hip } = spine;
  const neckBad = status.neckBad;
  const spineBad = status.spineBad;

  // bow amount = how much MORE forward than the calibrated baseline the head/shoulder
  // currently sit, not the raw absolute offset - otherwise the curve always bows forward
  // even in a good, calibrated posture (ears naturally sit slightly ahead of the shoulder)
  const torsoLength = distance(neck, hip) || 1;
  const neckForwardRatioNow = (head.x - neck.x) / torsoLength;
  const spineForwardRatioNow = (neck.x - hip.x) / torsoLength;
  const neckDeviation = baseline ? neckForwardRatioNow - baseline.neckForwardRatio : 0;
  const spineDeviationRaw = baseline ? spineForwardRatioNow - baseline.spineForwardRatio : 0;

  // the hip landmark is noisier than the shoulder/ear (often half out of frame at a desk),
  // so the spine bow gets its own smoothing to flow with the actual lean instead of jittering
  smoothedSpineDeviation =
    smoothedSpineDeviation === null
      ? spineDeviationRaw
      : smoothedSpineDeviation + (spineDeviationRaw - smoothedSpineDeviation) * SPINE_SMOOTHING_ALPHA;
  const spineDeviation = smoothedSpineDeviation;

  // neck curve: bows forward only as much as the neck currently deviates from baseline
  const neckMid = midpoint(head, neck);
  const neckControl = { x: neckMid.x + neckDeviation * torsoLength * NECK_BEND_GAIN, y: neckMid.y };
  drawCurveSegment(head, neckControl, neck, neckBad);

  // spine curve: bows forward only as much as the shoulder currently deviates from baseline
  const spineMid = midpoint(neck, hip);
  const spineControl = { x: spineMid.x + spineDeviation * torsoLength * SPINE_BEND_GAIN, y: spineMid.y };
  drawCurveSegment(neck, spineControl, hip, spineBad);

  const dots = [
    [head, neckBad],
    [neck, neckBad || spineBad],
    [hip, spineBad],
  ];
  for (const [p, isBad] of dots) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = isBad ? "#ff4d4d" : "#00e5ff";
    ctx.fill();
  }
}

function drawSkeleton(poseLandmarksList, handResult, spine, status, baseline) {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const drawingUtils = new DrawingUtils(ctx);

  for (const landmarks of poseLandmarksList) {
    drawingUtils.drawConnectors(landmarks, BODY_CONNECTIONS, { color: "#3d7cff", lineWidth: 2 });
    const bodyDots = landmarks.filter(
      (_, idx) => !HAND_POINT_INDICES.has(idx) && !FACE_EXTRA_INDICES.has(idx)
    );
    drawingUtils.drawLandmarks(bodyDots, { color: "#ffffff", radius: 2 });
  }

  if (handResult) {
    for (const landmarks of handResult.landmarks) {
      drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#ffd23d", lineWidth: 2 });
      drawingUtils.drawLandmarks(landmarks, { color: "#ffffff", radius: 2 });
    }
  }

  if (spine) drawSpineCurve(spine, status, baseline);

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
    drawSkeleton(poseResult.landmarks, handResult, null, NO_ISSUE, baselines[mode]);
    return;
  }

  const px = toPixelSpace(poseResult.landmarks[0]);
  const spine = computeSpinePoints(px);
  const metrics = mode === "front" ? computeMetricsFront(px) : computeMetricsSide(px);

  if (calibrating) {
    calibrationSamples.push(metrics);
    if (mode === "side") sideVotes.push(pickVisibleSide(px));
    drawSkeleton(poseResult.landmarks, handResult, spine, NO_ISSUE, baselines[mode]);
    return;
  }

  metricBuffer.push(metrics);
  if (metricBuffer.length > SMOOTHING_WINDOW) metricBuffer.shift();
  const smoothed = average(metricBuffer);
  const status = updateStatus(smoothed, now);

  drawSkeleton(poseResult.landmarks, handResult, spine, status, baselines[mode]);
}

async function runCalibration() {
  calibrating = true;
  calibrationSamples = [];
  sideVotes = [];
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
  if (mode === "side" && sideVotes.length > 0) {
    const leftCount = sideVotes.filter((s) => s === "left").length;
    lockedSide = leftCount >= sideVotes.length / 2 ? "left" : "right";
  }
  badSince = { neck: null, back: null, shoulder: null, waist: null };
  metricBuffer = [];
  smoothedSpineDeviation = null;
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
