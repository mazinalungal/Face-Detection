const VISION_VERSION = "1.1.1";
const VISION_MODULE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VISION_VERSION}/vision_bundle.mjs`;
const VISION_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VISION_VERSION}/wasm`;
const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite";

const video = document.querySelector("#camera-video");
const canvas = document.querySelector("#detection-canvas");
const context = canvas.getContext("2d");
const stage = document.querySelector("#video-stage");
const liveIndicator = document.querySelector(".preview-heading .live-indicator");
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const cameraSelect = document.querySelector("#camera-select");
const snapshotButton = document.querySelector("#snapshot-button");
const thresholdSlider = document.querySelector("#threshold-slider");
const thresholdValue = document.querySelector("#threshold-value");
const mirrorToggle = document.querySelector("#mirror-toggle");
const statusDot = document.querySelector("#status-dot");
const statusText = document.querySelector("#status-text");
const previewState = document.querySelector("#preview-state");
const emptyState = document.querySelector("#empty-state");
const resolutionLabel = document.querySelector("#resolution-label");
const fpsLabel = document.querySelector("#fps-label");
const faceCount = document.querySelector("#face-count");
const confidenceValue = document.querySelector("#confidence-value");
const errorMessage = document.querySelector("#error-message");

let faceDetector;
let cameraStream;
let animationFrameId;
let sessionId = 0;
let threshold = Number(thresholdSlider.value) / 100;
let lastVideoTime = -1;
let framesThisSecond = 0;
let fpsWindowStart = 0;
let latestDetections = [];
let rawDetections = [];

function setStatus(state, message) {
  const isLive = state === "live";
  statusDot.classList.toggle("is-live", isLive);
  statusText.textContent = message;
  previewState.textContent = state === "loading" ? "LOADING" : isLive ? "LIVE" : state === "error" ? "ERROR" : "STANDBY";
  previewState.classList.toggle("is-live", isLive);
  liveIndicator.classList.toggle("is-live", isLive);
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.hidden = true;
}

function stopTracks() {
  if (cameraStream) {
    for (const track of cameraStream.getTracks()) {
      track.stop();
    }
    cameraStream = undefined;
  }
  video.srcObject = null;
  video.classList.remove("is-visible");
  emptyState.hidden = false;
  snapshotButton.disabled = true;
  resolutionLabel.textContent = "— × —";
  clearOverlay();
  resetStatistics();
}

function clearOverlay() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  latestDetections = [];
  rawDetections = [];
}

function resetStatistics() {
  faceCount.textContent = "0";
  confidenceValue.textContent = "—";
  fpsLabel.innerHTML = '<span class="fps-dot"></span> — FPS';
  lastVideoTime = -1;
  framesThisSecond = 0;
  fpsWindowStart = 0;
}

async function loadDetector() {
  if (faceDetector) {
    return faceDetector;
  }

  const { FaceDetector, FilesetResolver } = await import(VISION_MODULE_URL);
  const fileset = await FilesetResolver.forVisionTasks(VISION_WASM_URL);
  faceDetector = await FaceDetector.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: FACE_MODEL_URL },
    runningMode: "VIDEO",
    minDetectionConfidence: 0.1,
  });
  return faceDetector;
}

async function listCameras(preferredDeviceId = cameraSelect.value) {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return;
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === "videoinput");
  cameraSelect.replaceChildren();

  if (cameras.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No cameras found";
    cameraSelect.append(option);
    cameraSelect.disabled = true;
    return;
  }

  cameras.forEach((camera, index) => {
    const option = document.createElement("option");
    option.value = camera.deviceId;
    option.textContent = camera.label || `Camera ${index + 1}`;
    cameraSelect.append(option);
  });

  cameraSelect.disabled = cameras.length < 2;
  if (preferredDeviceId && cameras.some((camera) => camera.deviceId === preferredDeviceId)) {
    cameraSelect.value = preferredDeviceId;
  }
}

function cameraConstraints(deviceId) {
  const videoConstraints = {
    width: { ideal: 1280 },
    height: { ideal: 800 },
    frameRate: { ideal: 30, max: 60 },
  };

  if (deviceId) {
    videoConstraints.deviceId = { exact: deviceId };
  }

  return { audio: false, video: videoConstraints };
}

async function connectCamera(activeSession, deviceId) {
  const stream = await navigator.mediaDevices.getUserMedia(cameraConstraints(deviceId));
  if (activeSession !== sessionId) {
    stream.getTracks().forEach((track) => track.stop());
    return;
  }

  cameraStream = stream;
  video.srcObject = stream;
  await video.play();
  if (activeSession !== sessionId) {
    return;
  }

  video.classList.add("is-visible");
  emptyState.hidden = true;
  snapshotButton.disabled = false;
  stage.classList.toggle("is-mirrored", mirrorToggle.checked);
  resolutionLabel.textContent = `${video.videoWidth} × ${video.videoHeight}`;
  setStatus("live", "Detection is running");
  stopButton.disabled = false;
  startButton.disabled = false;
  startButton.querySelector("span").textContent = "Restart detection";
  latestDetections = [];
  lastVideoTime = -1;
  fpsWindowStart = performance.now();
  animationFrameId = requestAnimationFrame(detectFrame);

  await listCameras(deviceId);
  if (activeSession !== sessionId) {
    return;
  }
}

async function startDetection() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("error", "Camera unavailable");
    showError("Camera access requires a modern browser on localhost or HTTPS.");
    return;
  }

  const activeSession = ++sessionId;
  clearError();
  cancelAnimationFrame(animationFrameId);
  stopTracks();
  startButton.disabled = true;
  stopButton.disabled = false;
  setStatus("loading", "Preparing face detector…");
  startButton.querySelector("span").textContent = "Loading detector…";

  let startupPhase = "detector";
  try {
    await loadDetector();
    if (activeSession !== sessionId) {
      return;
    }
    setStatus("loading", "Waiting for camera permission…");
    startupPhase = "camera";
    await connectCamera(activeSession, cameraSelect.value);
  } catch (error) {
    if (activeSession !== sessionId) {
      return;
    }
    stopTracks();
    stopButton.disabled = true;
    startButton.disabled = false;
    startButton.querySelector("span").textContent = "Start detection";
    cameraSelect.disabled = false;
    setStatus("error", "Could not start detection");
    showError(startupPhase === "detector" ? getDetectorError(error) : getCameraError(error));
  }
}

function stopDetection() {
  sessionId += 1;
  cancelAnimationFrame(animationFrameId);
  stopTracks();
  startButton.disabled = false;
  startButton.querySelector("span").textContent = "Start detection";
  stopButton.disabled = true;
  cameraSelect.disabled = false;
  clearError();
  setStatus("idle", "Camera is off");
}

async function changeCamera() {
  if (!cameraStream) {
    return;
  }

  const activeSession = ++sessionId;
  cancelAnimationFrame(animationFrameId);
  stopTracks();
  clearError();
  setStatus("loading", "Switching camera…");
  startButton.disabled = true;
  stopButton.disabled = false;

  try {
    await connectCamera(activeSession, cameraSelect.value);
  } catch (error) {
    if (activeSession !== sessionId) {
      return;
    }
    stopTracks();
    startButton.disabled = false;
    startButton.querySelector("span").textContent = "Start detection";
    stopButton.disabled = true;
    cameraSelect.disabled = false;
    setStatus("error", "Could not switch camera");
    showError(getCameraError(error));
  }
}

function getDetectorError(error) {
  const details = error instanceof Error && error.message ? ` Details: ${error.message}` : "";
  return `The face detector could not be loaded. Check your internet connection and reload the page.${details}`;
}

function getCameraError(error) {
  if (error?.name === "NotAllowedError" || error?.name === "PermissionDeniedError") {
    return "Camera permission was denied. Allow camera access in your browser settings, then try again.";
  }
  if (error?.name === "NotFoundError" || error?.name === "DevicesNotFoundError") {
    return "No camera was found. Connect a camera and try again.";
  }
  if (error?.name === "NotReadableError" || error?.name === "TrackStartError") {
    return "The camera is already in use or could not be started. Close other apps using it and try again.";
  }
  if (error instanceof TypeError && !navigator.mediaDevices?.getUserMedia) {
    return "Camera access requires a modern browser on localhost or HTTPS.";
  }
  if (error instanceof TypeError && error.message) {
    return `The camera could not be opened. Check the selected camera and browser permissions. Details: ${error.message}`;
  }
  if (error instanceof Error && error.message) {
    return `The camera could not be opened. Details: ${error.message}`;
  }
  return "The camera could not be opened. Check your camera connection and browser permissions, then try again.";
}

function getDetectionError(error) {
  const details = error instanceof Error && error.message ? ` Details: ${error.message}` : "";
  return `Face detection stopped unexpectedly.${details}`;
}

function resizeOverlay() {
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    return;
  }

  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
}

function getScore(detection) {
  return detection.categories?.[0]?.score ?? 0;
}

function detectFrame(now) {
  if (!cameraStream || !faceDetector || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    animationFrameId = requestAnimationFrame(detectFrame);
    return;
  }

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    try {
      resizeOverlay();
      const results = faceDetector.detectForVideo(video, now);
      rawDetections = results.detections;
      latestDetections = rawDetections.filter((detection) => getScore(detection) >= threshold);
      drawDetections(latestDetections);
      updateStatistics(latestDetections, now);
    } catch (error) {
      sessionId += 1;
      cancelAnimationFrame(animationFrameId);
      stopTracks();
      startButton.disabled = false;
      startButton.querySelector("span").textContent = "Start detection";
      stopButton.disabled = true;
      setStatus("error", "Detection stopped unexpectedly");
      showError(getDetectionError(error));
      return;
    }
  }

  animationFrameId = requestAnimationFrame(detectFrame);
}

function drawDetections(detections) {
  context.clearRect(0, 0, canvas.width, canvas.height);
  for (const detection of detections) {
    const box = detection.boundingBox;
    if (!box) {
      continue;
    }

    context.strokeStyle = "#73edc3";
    context.lineWidth = Math.max(2, canvas.width / 500);
    context.strokeRect(box.originX, box.originY, box.width, box.height);

    const label = `FACE  ${Math.round(getScore(detection) * 100)}%`;
    const fontSize = Math.max(14, canvas.width / 90);
    context.font = `500 ${fontSize}px "DM Mono", monospace`;
    const labelWidth = context.measureText(label).width + 16;
    const labelHeight = fontSize + 12;
    const labelX = Math.max(0, box.originX);
    const labelY = Math.max(labelHeight, box.originY);

    context.fillStyle = "#73edc3";
    context.fillRect(labelX, labelY - labelHeight, labelWidth, labelHeight);
    context.fillStyle = "#10211f";
    context.fillText(label, labelX + 8, labelY - 7);
  }
}

function updateStatistics(detections, now) {
  updateFaceStats(detections);

  framesThisSecond += 1;
  if (now - fpsWindowStart >= 1000) {
    fpsLabel.innerHTML = `<span class="fps-dot"></span> ${Math.round((framesThisSecond * 1000) / (now - fpsWindowStart))} FPS`;
    framesThisSecond = 0;
    fpsWindowStart = now;
  }
}

function updateFaceStats(detections) {
  faceCount.textContent = String(detections.length);
  confidenceValue.textContent = detections.length
    ? `${Math.round((detections.reduce((sum, detection) => sum + getScore(detection), 0) / detections.length) * 100)}%`
    : "—";
}

function updateThreshold() {
  threshold = Number(thresholdSlider.value) / 100;
  thresholdValue.value = `${thresholdSlider.value}%`;
  thresholdSlider.style.background = `linear-gradient(90deg, var(--mint) 0%, var(--mint) ${thresholdSlider.value}%, #2a344b ${thresholdSlider.value}%, #2a344b 100%)`;

  if (cameraStream) {
    latestDetections = rawDetections.filter((detection) => getScore(detection) >= threshold);
    drawDetections(latestDetections);
    updateFaceStats(latestDetections);
  }
}

function saveSnapshot() {
  if (!cameraStream || video.videoWidth === 0) {
    return;
  }

  const snapshot = document.createElement("canvas");
  snapshot.width = video.videoWidth;
  snapshot.height = video.videoHeight;
  const snapshotContext = snapshot.getContext("2d");

  if (mirrorToggle.checked) {
    snapshotContext.translate(snapshot.width, 0);
    snapshotContext.scale(-1, 1);
  }
  snapshotContext.drawImage(video, 0, 0, snapshot.width, snapshot.height);
  snapshotContext.drawImage(canvas, 0, 0);

  snapshot.toBlob((blob) => {
    if (!blob) {
      showError("The snapshot could not be created. Please try again.");
      return;
    }
    const downloadUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = `facescope-${new Date().toISOString().replace(/[:.]/g, "-")}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);
  }, "image/png");
}

startButton.addEventListener("click", startDetection);
stopButton.addEventListener("click", stopDetection);
cameraSelect.addEventListener("change", changeCamera);
snapshotButton.addEventListener("click", saveSnapshot);
thresholdSlider.addEventListener("input", updateThreshold);
mirrorToggle.addEventListener("change", () => {
  stage.classList.toggle("is-mirrored", mirrorToggle.checked);
});

listCameras().catch(() => {
  cameraSelect.disabled = true;
});
updateThreshold();
