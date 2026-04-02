const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);
var useWebGPU = urlParams.get('webgpu') !== '0';

const { AutoModel, AutoProcessor, RawImage, Tensor, env } = window.HF;

if (!navigator.gpu) {
  const errorMessage = "WebGPU is not supported by this browser.";
  throw alert(errorMessage), Error(errorMessage);
}

env.backends.onnx.wasm.proxy = true;
env.backends.onnx.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";
env.backends.onnx.wasm.numThreads = 1;
env.experimental.useWebGPU = useWebGPU;

const DEFAULT_IMAGE_URL = "hikari.jpeg";
const statusElement = document.getElementById("status");
const fileInputElement = document.getElementById("upload");
const imageContainerElement = document.getElementById("container");
const pasteCanvas = document.getElementById("paste-canvas");
const resetButton = document.getElementById("reset-button");
const selectAreaButton = document.getElementById("select-area-button");
const selectCanvas = document.getElementById("select-canvas");
const colorPicker = document.getElementById("color-picker");
const selectCtx = selectCanvas.getContext("2d");
const uploadButtonHtml = imageContainerElement.innerHTML;
const exampleButtonSelector = "#example";

let pasteImage = null;
let pastePosition = { x: 0, y: 0 };
let pasteDrag = { active: false, startX: 0, startY: 0, imageX: 0, imageY: 0 };
let rawImage = null;
let currentImageSource = null;
let isSelectingArea = false;
let selectRectActive = false;
let selectRect = { startX: 0, startY: 0, endX: 0, endY: 0 };
let displayWidth = 0;
let displayHeight = 0;

const pasteCtx = pasteCanvas.getContext("2d");

function drawPasteCanvas() {
  if (!pasteImage) {
    pasteCtx.clearRect(0, 0, pasteCanvas.width, pasteCanvas.height);
    return;
  }

  pasteCtx.clearRect(0, 0, pasteCanvas.width, pasteCanvas.height);
  pasteCtx.fillStyle = colorPicker.value;
  pasteCtx.fillRect(0, 0, pasteCanvas.width, pasteCanvas.height);

  const imageRatio = pasteImage.width / pasteImage.height;
  const canvasRatio = pasteCanvas.width / pasteCanvas.height;
  let drawWidth = pasteCanvas.width;
  let drawHeight = pasteCanvas.height;

  if (imageRatio > canvasRatio) {
    drawHeight = pasteCanvas.width / imageRatio;
  } else {
    drawWidth = pasteCanvas.height * imageRatio;
  }

  const drawX = pastePosition.x || (pasteCanvas.width - drawWidth) / 2;
  const drawY = pastePosition.y || (pasteCanvas.height - drawHeight) / 2;

  pasteCtx.drawImage(pasteImage, drawX, drawY, drawWidth, drawHeight);
}

async function copyCanvasToClipboard(canvas) {
  if (!navigator.clipboard?.write) return;
  try {
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
    if (!blob) return;
    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
  } catch (error) {
    console.warn("Clipboard copy failed:", error);
  }
}

pasteCanvas.addEventListener("pointerdown", event => {
  if (!pasteImage) return;
  const rect = pasteCanvas.getBoundingClientRect();
  const scaleX = pasteCanvas.width / rect.width;
  const scaleY = pasteCanvas.height / rect.height;
  pasteDrag.active = true;
  pasteDrag.startX = (event.clientX - rect.left) * scaleX;
  pasteDrag.startY = (event.clientY - rect.top) * scaleY;
  pasteDrag.imageX = pastePosition.x;
  pasteDrag.imageY = pastePosition.y;
  pasteCanvas.setPointerCapture(event.pointerId);
  pasteCanvas.classList.add("dragging");
});

pasteCanvas.addEventListener("pointermove", event => {
  if (!pasteDrag.active) return;
  const rect = pasteCanvas.getBoundingClientRect();
  const scaleX = pasteCanvas.width / rect.width;
  const scaleY = pasteCanvas.height / rect.height;
  const currentX = (event.clientX - rect.left) * scaleX;
  const currentY = (event.clientY - rect.top) * scaleY;
  pastePosition.x = pasteDrag.imageX + (currentX - pasteDrag.startX);
  pastePosition.y = pasteDrag.imageY + (currentY - pasteDrag.startY);
  drawPasteCanvas();
});

pasteCanvas.addEventListener("pointerup", event => {
  if (!pasteDrag.active) return;
  pasteDrag.active = false;
  pasteCanvas.releasePointerCapture(event.pointerId);
  pasteCanvas.classList.remove("dragging");
});

pasteCanvas.addEventListener("pointerleave", () => {
  if (!pasteDrag.active) return;
  pasteDrag.active = false;
  pasteCanvas.classList.remove("dragging");
});

statusElement.textContent = "Loading model...";

let backgroundRemovalModel;

try {
  backgroundRemovalModel = await AutoModel.from_pretrained("briaai/RMBG-1.4", {
    config: {
      model_type: "custom"
    },
    quantized: false
  });
} catch (error) {
  statusElement.textContent = error.message;
  alert(error.message);
  throw error;
}

const imageProcessor = await AutoProcessor.from_pretrained("briaai/RMBG-1.4", {
  config: {
    do_normalize: true,
    do_pad: false,
    do_rescale: true,
    do_resize: true,
    image_mean: [0.5, 0.5, 0.5],
    feature_extractor_type: "ImageFeatureExtractor",
    image_std: [1, 1, 1],
    resample: 2,
    rescale_factor: 0.00392156862745098,
    size: {
      width: 1024,
      height: 1024
    }
  }
});

statusElement.textContent = "Warming up...";
const [batchSize, channelCount, imageHeight, imageWidth] = [1, 3, 1024, 1024];
const warmupTensorData = new Float32Array(batchSize * channelCount * imageHeight * imageWidth);
await backgroundRemovalModel({
  input: new Tensor("float32", warmupTensorData, [batchSize, channelCount, imageHeight, imageWidth])
});
statusElement.textContent = "Ready";

imageContainerElement.addEventListener("click", event => {
  const target = event.target.closest(exampleButtonSelector);
  if (!target) return;
  event.preventDefault();
  runBackgroundRemoval(DEFAULT_IMAGE_URL);
});

fileInputElement.addEventListener("change", function (event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = loadEvent => runBackgroundRemoval(loadEvent.target.result);
  reader.readAsDataURL(file);
});

resetButton.addEventListener("click", () => {
  fileInputElement.value = "";
  imageContainerElement.innerHTML = uploadButtonHtml;
  imageContainerElement.appendChild(selectCanvas); // giữ select canvas khi reset
  imageContainerElement.style.removeProperty("background-image");
  imageContainerElement.style.removeProperty("background");
  imageContainerElement.style.removeProperty("width");
  imageContainerElement.style.removeProperty("height");
  pasteImage = null;
  rawImage = null;
  isSelectingArea = false;
  selectRectActive = false;
  selectRect = { startX: 0, startY: 0, endX: 0, endY: 0 };
  displayWidth = 0;
  displayHeight = 0;
  selectCanvas.style.display = "none";
  selectCtx.clearRect(0, 0, selectCanvas.width, selectCanvas.height);
  drawPasteCanvas();
  statusElement.textContent = "Select a new image to continue.";
});

selectAreaButton.addEventListener("click", () => {
  if (!rawImage) {
    alert("Please upload an image first.");
    return;
  }
  isSelectingArea = true;
  selectCanvas.style.display = "block";
  selectCanvas.style.pointerEvents = "auto";
  statusElement.textContent = "Select an area by dragging the mouse.";
});

selectCanvas.addEventListener("pointerdown", event => {
  if (!isSelectingArea) return;
  const rect = selectCanvas.getBoundingClientRect();
  const scaleX = selectCanvas.width / rect.width;
  const scaleY = selectCanvas.height / rect.height;
  selectRect.startX = (event.clientX - rect.left) * scaleX;
  selectRect.startY = (event.clientY - rect.top) * scaleY;
  selectRect.endX = selectRect.startX;
  selectRect.endY = selectRect.startY;
  selectRectActive = true;
  selectCanvas.setPointerCapture(event.pointerId);
});

selectCanvas.addEventListener("pointermove", event => {
  if (!isSelectingArea || !selectRectActive) return;
  const rect = selectCanvas.getBoundingClientRect();
  const scaleX = selectCanvas.width / rect.width;
  const scaleY = selectCanvas.height / rect.height;
  selectRect.endX = (event.clientX - rect.left) * scaleX;
  selectRect.endY = (event.clientY - rect.top) * scaleY;
  drawSelectRect();
});

colorPicker.addEventListener("change", function(event) {
  drawPasteCanvas();
});

selectCanvas.addEventListener("pointerup", async event => {
  if (!isSelectingArea) return;
  selectCanvas.releasePointerCapture(event.pointerId);
  isSelectingArea = false;
  selectRectActive = false;
  selectCanvas.style.display = "none";
  selectCanvas.style.pointerEvents = "none";
  selectCtx.clearRect(0, 0, selectCanvas.width, selectCanvas.height);
  await processSelectedArea();
});

function drawSelectRect() {
  selectCtx.clearRect(0, 0, selectCanvas.width, selectCanvas.height);
  selectCtx.strokeStyle = "#ff0000";
  selectCtx.lineWidth = 2;
  selectCtx.setLineDash([5, 5]);
  selectCtx.strokeRect(
    Math.min(selectRect.startX, selectRect.endX),
    Math.min(selectRect.startY, selectRect.endY),
    Math.abs(selectRect.endX - selectRect.startX),
    Math.abs(selectRect.endY - selectRect.startY)
  );
}

async function processSelectedArea() {
  const scaleX = displayWidth / rawImage.width;
  const scaleY = displayHeight / rawImage.height;

  const cropX = Math.min(selectRect.startX, selectRect.endX) / scaleX;
  const cropY = Math.min(selectRect.startY, selectRect.endY) / scaleY;
  const cropWidth = Math.abs(selectRect.endX - selectRect.startX) / scaleX;
  const cropHeight = Math.abs(selectRect.endY - selectRect.startY) / scaleY;

  if (cropWidth < 1 || cropHeight < 1) {
    statusElement.textContent = "Selected area too small.";
    return;
  }

  // Clamp and round to integer coords for RawImage.crop ([left, top, right, bottom])
  const left = Math.max(0, Math.round(cropX));
  const top = Math.max(0, Math.round(cropY));
  const right = Math.min(rawImage.width - 1, Math.round(cropX + cropWidth - 1));
  const bottom = Math.min(rawImage.height - 1, Math.round(cropY + cropHeight - 1));

  if (right <= left || bottom <= top) {
    statusElement.textContent = "Selected area too small or invalid.";
    return;
  }

  // Crop the rawImage with the correct argument format
  const croppedImage = await rawImage.crop([left, top, right, bottom]);

  statusElement.textContent = "Processing selected area...";

  // Process the cropped image
  const { pixel_values: pixelValues } = await imageProcessor(croppedImage);
  const inferenceStartTime = performance.now();
  const { output: modelOutput } = await backgroundRemovalModel({
    input: pixelValues
  });
  const inferenceEndTime = performance.now();

  const maskImage = await RawImage.fromTensor(modelOutput[0].mul(255).to("uint8")).resize(croppedImage.width, croppedImage.height);

  // Create result canvas for cropped area
  const resultCanvas = document.createElement("canvas");
  resultCanvas.width = croppedImage.width;
  resultCanvas.height = croppedImage.height;

  const canvasContext = resultCanvas.getContext("2d");
  canvasContext.drawImage(croppedImage.toCanvas(), 0, 0);

  const imageData = canvasContext.getImageData(0, 0, croppedImage.width, croppedImage.height);
  for (let pixelIndex = 0; pixelIndex < maskImage.data.length; ++pixelIndex) {
    imageData.data[4 * pixelIndex + 3] = maskImage.data[pixelIndex];
  }

  canvasContext.putImageData(imageData, 0, 0);

  // Không ghi đè ảnh crop lên container đang chọn; chỉ cập nhật paste canvas
  imageContainerElement.style.backgroundImage = currentImageSource ? `url(${currentImageSource})` : imageContainerElement.style.backgroundImage;
  imageContainerElement.style.backgroundSize = "100% 100%";
  imageContainerElement.style.backgroundPosition = "center";

  // show result in paste-canvas (giữ canvas hiển thị đúng ratio)
  pasteCanvas.width = pasteCanvas.clientWidth;
  pasteCanvas.height = pasteCanvas.clientHeight;
  pasteImage = new Image();
  pasteImage.src = resultCanvas.toDataURL("image/png");
  pastePosition = { x: 0, y: 0 };
  pasteImage.onload = () => {
    drawPasteCanvas();
  };

  await copyCanvasToClipboard(resultCanvas);
  statusElement.textContent = `Done! (Inference took ${Math.round(inferenceEndTime - inferenceStartTime)}ms)`;
}

async function runBackgroundRemoval(imageSource) {
  const localRawImage = await RawImage.fromURL(imageSource);

  // Giữ selectCanvas (và #upload-button nếu còn) – chỉ reset image cũ
  const oldResult = imageContainerElement.querySelector("canvas.result-canvas");
  if (oldResult) oldResult.remove();
  const uploadButton = imageContainerElement.querySelector("#upload-button");
  if (uploadButton) uploadButton.remove();

  currentImageSource = imageSource;
  imageContainerElement.style.backgroundImage = `url(${currentImageSource})`;
  imageContainerElement.style.backgroundSize = "100% 100%";
  imageContainerElement.style.backgroundPosition = "center";

  const aspectRatio = localRawImage.width / localRawImage.height;
  let newDisplayWidth, newDisplayHeight;
  if (aspectRatio > 720 / 480) {
    newDisplayWidth = 720;
    newDisplayHeight = 720 / aspectRatio;
  } else {
    newDisplayWidth = 480 * aspectRatio;
    newDisplayHeight = 480;
  }
  imageContainerElement.style.width = `${newDisplayWidth}px`;
  imageContainerElement.style.height = `${newDisplayHeight}px`;
  statusElement.textContent = "Analysing...";

  // Store for select area
  rawImage = localRawImage;
  displayWidth = newDisplayWidth;
  displayHeight = newDisplayHeight;
  selectCanvas.width = displayWidth;
  selectCanvas.height = displayHeight;

  const { pixel_values: pixelValues } = await imageProcessor(localRawImage);
  const inferenceStartTime = performance.now();
  const { output: modelOutput } = await backgroundRemovalModel({
    input: pixelValues
  });
  const inferenceEndTime = performance.now();

  const maskImage = await RawImage.fromTensor(modelOutput[0].mul(255).to("uint8")).resize(localRawImage.width, localRawImage.height);
  const resultCanvas = document.createElement("canvas");
  resultCanvas.className = "result-canvas";
  resultCanvas.width = localRawImage.width;
  resultCanvas.height = localRawImage.height;

  const canvasContext = resultCanvas.getContext("2d");
  canvasContext.drawImage(localRawImage.toCanvas(), 0, 0);

  const imageData = canvasContext.getImageData(0, 0, localRawImage.width, localRawImage.height);
  for (let pixelIndex = 0; pixelIndex < maskImage.data.length; ++pixelIndex) {
    imageData.data[4 * pixelIndex + 3] = maskImage.data[pixelIndex];
  }

  canvasContext.putImageData(imageData, 0, 0);
  imageContainerElement.append(resultCanvas);
  imageContainerElement.style.removeProperty("background-image");
  imageContainerElement.style.background = 'url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAGUExURb+/v////5nD/3QAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAUSURBVBjTYwABQSCglEENMxgYGAAynwRB8BEAgQAAAABJRU5ErkJggg==")';

  pasteCanvas.width = pasteCanvas.clientWidth;
  pasteCanvas.height = pasteCanvas.clientHeight;
  pasteImage = new Image();
  pasteImage.src = resultCanvas.toDataURL("image/png");
  pastePosition = { x: 0, y: 0 };
  pasteImage.onload = () => {
    drawPasteCanvas();
  };

  await copyCanvasToClipboard(resultCanvas);
  statusElement.textContent = `Done! (Inference took ${Math.round(inferenceEndTime - inferenceStartTime)}ms)`;
}
