const { AutoModel, AutoProcessor, RawImage, Tensor, env } = window.HF;

if (!navigator.gpu) {
  const errorMessage = "WebGPU is not supported by this browser.";
  throw alert(errorMessage), Error(errorMessage);
}

env.backends.onnx.wasm.proxy = true;
env.backends.onnx.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";
env.backends.onnx.wasm.numThreads = 1;
env.experimental.useWebGPU = true;

const DEFAULT_IMAGE_URL = "hikari.jpeg";
const statusElement = document.getElementById("status");
const fileInputElement = document.getElementById("upload");
const imageContainerElement = document.getElementById("container");
const pasteCanvas = document.getElementById("paste-canvas");
const resetButton = document.getElementById("reset-button");
const uploadButtonHtml = imageContainerElement.innerHTML;
const exampleButtonSelector = "#example";

let pasteImage = null;
let pastePosition = { x: 0, y: 0 };
let pasteDrag = { active: false, startX: 0, startY: 0, imageX: 0, imageY: 0 };

const pasteCtx = pasteCanvas.getContext("2d");

function drawPasteCanvas() {
  if (!pasteImage) {
    pasteCtx.clearRect(0, 0, pasteCanvas.width, pasteCanvas.height);
    return;
  }

  pasteCtx.clearRect(0, 0, pasteCanvas.width, pasteCanvas.height);
  pasteCtx.fillStyle = "#000";
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
  imageContainerElement.style.removeProperty("background-image");
  imageContainerElement.style.removeProperty("background");
  imageContainerElement.style.removeProperty("width");
  imageContainerElement.style.removeProperty("height");
  pasteImage = null;
  drawPasteCanvas();
  statusElement.textContent = "Select a new image to continue.";
});

async function runBackgroundRemoval(imageSource) {
  const rawImage = await RawImage.fromURL(imageSource);

  imageContainerElement.innerHTML = "";
  imageContainerElement.style.backgroundImage = `url(${imageSource})`;

  const aspectRatio = rawImage.width / rawImage.height;
  const [displayWidth, displayHeight] = aspectRatio > 720 / 480 ? [720, 720 / aspectRatio] : [480 * aspectRatio, 480];
  imageContainerElement.style.width = `${displayWidth}px`;
  imageContainerElement.style.height = `${displayHeight}px`;
  statusElement.textContent = "Analysing...";

  const { pixel_values: pixelValues } = await imageProcessor(rawImage);
  const inferenceStartTime = performance.now();
  const { output: modelOutput } = await backgroundRemovalModel({
    input: pixelValues
  });
  const inferenceEndTime = performance.now();

  const maskImage = await RawImage.fromTensor(modelOutput[0].mul(255).to("uint8")).resize(rawImage.width, rawImage.height);
  const resultCanvas = document.createElement("canvas");
  resultCanvas.width = rawImage.width;
  resultCanvas.height = rawImage.height;

  const canvasContext = resultCanvas.getContext("2d");
  canvasContext.drawImage(rawImage.toCanvas(), 0, 0);

  const imageData = canvasContext.getImageData(0, 0, rawImage.width, rawImage.height);
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
