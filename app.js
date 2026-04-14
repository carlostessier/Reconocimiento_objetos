import {
  env,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

// --------------------------------------------------------------------------------
// CONFIGURACIÓN DE MODELOS LOCALES (OFFLINE / GITHUB PAGES)
// --------------------------------------------------------------------------------

// Paso 1: Desactivar la descarga desde internet (Hugging Face)
env.allowRemoteModels = false;

// Paso 2: Indicar a la librería que busque los archivos en nuestra carpeta "models/"
// De esta forma los descargará de nuestro propio GitHub Pages.
env.localModelPath = './models/';

// Es REQUISITO que esté a true si desactivas la descarga remota (allowRemoteModels).
// Al ejecutarse en un entorno web, la librería sabe interpretar automáticamente
// que debe usar peticiones "fetch" hacia tu host en lugar de buscar en un disco duro.
env.allowLocalModels = true;

// --------------------------------------------------------------------------------

// Configuración general
const MODEL_ID = "onnx-community/rfdetr_nano-ONNX";
const DETECTION_INTERVAL_MS = 450; // Tiempo entre cada detección en la cámara
const MAX_VISIBLE_DETECTIONS = 8; // Máximo de objetos a mostrar simultáneamente

// Referencias a los elementos de HTML (Botones, selectores, video y canvas)
const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const cameraSelect = document.getElementById("camera-select");
const thresholdInput = document.getElementById("threshold-input");
const thresholdOutput = document.getElementById("threshold-output");
const statusLine = document.getElementById("status-line");
const modelName = document.getElementById("model-name");
const backendName = document.getElementById("backend-name");
const runtimeState = document.getElementById("runtime-state");
const video = document.getElementById("webcam-video");
const overlayCanvas = document.getElementById("overlay-canvas");
const emptyState = document.getElementById("empty-state");

// Contexto 2D para dibujar en el canvas sobre el video
const overlayContext = overlayCanvas.getContext("2d");
// Canvas oculto auxiliar para procesar los frames del video
const inferenceCanvas = document.createElement("canvas");

// Variables de estado de la aplicación
let detector = null; // Guardará la tubería del modelo
let detectorBackend = "wasm"; // Usará WebAssembly o WebGPU
let mediaStream = null; // Stream actual de la cámara
let inferLoopTimer = null; // Temporizador del bucle de inferencia
let inferenceInFlight = false; // Flag para saber si se está procesando un frame
let latestDetections = []; // Últimos objetos detectados
let availableDevices = []; // Cámaras disponibles
let currentThreshold = Number(thresholdInput.value) / 100; // Umbral de confianza
let lastCaptureWidth = 1;
let lastCaptureHeight = 1;

// Inicializamos la información estática en la UI
modelName.textContent = MODEL_ID;
thresholdOutput.value = currentThreshold.toFixed(2);
thresholdOutput.textContent = currentThreshold.toFixed(2);

// ==========================================
// EVENTOS DE LA INTERFAZ
// ==========================================

// Actualizar el umbral en tiempo real con la barra deslizante
thresholdInput.addEventListener("input", () => {
  currentThreshold = Number(thresholdInput.value) / 100;
  thresholdOutput.value = currentThreshold.toFixed(2);
  thresholdOutput.textContent = currentThreshold.toFixed(2);
  drawDetections(latestDetections);
});

// Cambiar de cámara cuando se selecciona otra en la lista
cameraSelect.addEventListener("change", async () => {
  if (!cameraSelect.value) return;
  await startCamera(cameraSelect.value);
});

// Botón: Iniciar cámara
startButton.addEventListener("click", async () => {
  startButton.disabled = true;
  try {
    await ensureDetector(); // 1. Cargar el modelo IA
    await populateCameraOptions(); // 2. Obtener lista de cámaras
    // 3. Encender la cámara
    const selectedId = cameraSelect.value || availableDevices[0]?.deviceId || "";
    await startCamera(selectedId);
  } catch (error) {
    handleRuntimeError(error);
    startButton.disabled = false;
  }
});

// Botón: Detener cámara
stopButton.addEventListener("click", () => {
  stopDetectionLoop(); // Parar el escaneo continuo
  stopStream(); // Apagar la cámara física
  clearOverlay(); // Limpiar dibujos actuales
  latestDetections = [];
  
  // Refrescar UI visualmente
  runtimeState.textContent = "Detenido";
  statusLine.textContent = "Cámara detenida.";
  emptyState.hidden = false;
  startButton.disabled = false;
  stopButton.disabled = true;
  cameraSelect.disabled = availableDevices.length === 0;
});

// Ajustar dibujo cuando el video cambia de tamaño o se carga
video.addEventListener("loadedmetadata", () => {
  syncCanvasSize();
  drawDetections(latestDetections);
});

// Ajustar dibujo al cambiar tamaño de la ventana
window.addEventListener("resize", () => {
  syncCanvasSize();
  drawDetections(latestDetections);
});

// ==========================================
// FUNCIONES PRINCIPALES
// ==========================================

/**
 * Carga el modelo de detección de objetos.
 * Usa WebGPU si está disponible como hardware de aceleración gráfica; si no,
 * cambia al uso de la CPU con WASM (WebAssembly).
 */
async function ensureDetector() {
  if (detector) return detector;

  statusLine.textContent = "Cargando modelo de detección...";
  runtimeState.textContent = "Inicializando";

  // Evaluar aceleración disponible en el navegador
  detectorBackend = "webgpu" in navigator ? "webgpu" : "wasm";
  backendName.textContent = detectorBackend.toUpperCase();

  try {
    // Inicializar Pipeline usando la librería Transformers.js
    detector = await pipeline("object-detection", MODEL_ID, {
      device: detectorBackend,
      dtype: detectorBackend === "webgpu" ? "fp16" : "q8",
    });
  } catch (error) {
    // Mecanismo de emergencia "Fallback": Si WebGPU falla, vuelve a intentarlo bajando a CPU (WASM)
    if (detectorBackend !== "wasm") {
      detectorBackend = "wasm";
      backendName.textContent = "WASM";
      statusLine.textContent = "WebGPU no estuvo disponible; cambiando a WASM...";

      detector = await pipeline("object-detection", MODEL_ID, {
        device: "wasm",
        dtype: "q8",
      });
    } else {
      throw error;
    }
  }

  statusLine.textContent = "Modelo cargado. Listo para abrir la cámara.";
  runtimeState.textContent = "Modelo listo";
  return detector;
}

/**
 * Solicita los permisos de cámara y averigua cuántas cámaras tiene conectadas
 * el usuario para rellenar la lista desplegable de la UI.
 */
async function populateCameraOptions() {
  const preliminaryStream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: false,
  });

  preliminaryStream.getTracks().forEach((track) => track.stop());

  const devices = await navigator.mediaDevices.enumerateDevices();
  availableDevices = devices.filter((device) => device.kind === "videoinput");

  cameraSelect.innerHTML = "";

  if (availableDevices.length === 0) {
    cameraSelect.disabled = true;
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No se encontraron cámaras";
    cameraSelect.appendChild(option);
    return;
  }

  for (const [index, device] of availableDevices.entries()) {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `Cámara ${index + 1}`;
    cameraSelect.appendChild(option);
  }

  cameraSelect.disabled = false;
}

/**
 * Activa la transmisión de vídeo de la cámara que hemos elegido 
 * y comienza los ciclos continuos de inferencia a través del temporizador.
 */
async function startCamera(deviceId) {
  stopDetectionLoop();
  stopStream();

  statusLine.textContent = "Abriendo cámara...";
  runtimeState.textContent = "Solicitando permisos";

  const constraints = {
    audio: false,
    video: deviceId
      ? {
          deviceId: { exact: deviceId },
          facingMode: "environment", // Prioriza la cámara trasera en dispositivos móviles
          width: { ideal: 1280 },
          height: { ideal: 720 },
        }
      : {
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
  };

  mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = mediaStream;
  await video.play();

  // Seleccionar en el desplegable la cámara usada realmente
  const activeTrack = mediaStream.getVideoTracks()[0];
  const activeSettings = activeTrack?.getSettings();
  if (activeSettings?.deviceId) {
    cameraSelect.value = activeSettings.deviceId;
  }

  // Refrescar panel lateral de la UI
  emptyState.hidden = true;
  stopButton.disabled = false;
  startButton.disabled = true;
  runtimeState.textContent = "Detectando";
  statusLine.textContent = "Cámara activa. Analizando la escena cada pocos cientos de milisegundos.";

  // Configurar e iniciar inferencia continua  
  syncCanvasSize();
  runDetectionCycle(); // Arranca el primer frame manualmente
  inferLoopTimer = window.setInterval(runDetectionCycle, DETECTION_INTERVAL_MS); // Programa sus repeticiones
}

/**
 * Analiza el frame actual del vídeo usando el modelo de inteligencia artificial.
 * Se ejecuta repetidamente en bucle según el DETECTION_INTERVAL_MS definido.
 */
async function runDetectionCycle() {
  // Evitar solapamientos si una estimación se retrasa o si el modelo falló
  if (!detector || !mediaStream || inferenceInFlight || video.readyState < 2) {
    return;
  }

  inferenceInFlight = true;
  runtimeState.textContent = "Analizando";

  try {
    // Definir unas reducciones de imagen proporcionales óptimas para no congestionar
    // el rendimiento (es innecesario enviar 1080p a un modelo tan pequeño)
    const captureWidth = Math.max(320, Math.min(video.videoWidth, 960));
    const captureHeight = Math.max(
      180,
      Math.round((captureWidth / Math.max(video.videoWidth, 1)) * video.videoHeight),
    );

    inferenceCanvas.width = captureWidth;
    inferenceCanvas.height = captureHeight;
    lastCaptureWidth = captureWidth;
    lastCaptureHeight = captureHeight;

    const inferenceContext = inferenceCanvas.getContext("2d", { willReadFrequently: true });

    // Extraer un fotograma plano del vídeo actual pasándolo a JPEG
    inferenceContext.drawImage(video, 0, 0, captureWidth, captureHeight);
    const frameDataUrl = inferenceCanvas.toDataURL("image/jpeg", 0.82);

    // Enviar foto a la red neuronal y pedir que busque cajas superior al Umbral
    const results = await detector(frameDataUrl, {
      threshold: currentThreshold,
    });

    latestDetections = results.slice(0, MAX_VISIBLE_DETECTIONS);
    
    // Reflejar la actualización pintando las cajas delimintadoras
    drawDetections(latestDetections);
    
    statusLine.textContent =
      latestDetections.length > 0
        ? `Detecciones activas: ${latestDetections.length}`
        : "Sin detecciones claras en este frame.";
    runtimeState.textContent = "Detectando";
  } catch (error) {
    console.error(error);
    statusLine.textContent = "Error durante la inferencia. Reintentando...";
    runtimeState.textContent = "Reintentando";
  } finally {
    inferenceInFlight = false; // Marcar como listo para seguir procesando frames frescos
  }
}

/**
 * Pinta de colores los recuadros delimitadores (Bounding Boxes) devueltos por el detector
 * utilizando la API nativa de Canvas de HTML5.
 */
function drawDetections(detections) {
  syncCanvasSize();
  clearOverlay();

  // Paleta de colores atractiva para marcar los objetos cíclicamente
  const colors = [
    "#0e8d78", "#f97316", "#2563eb", "#be185d", 
    "#7c3aed", "#ca8a04", "#0891b2", "#059669",
  ];

  detections.forEach((entry, index) => {
    const color = colors[index % colors.length];
    const { box, label, score } = entry;
    
    // Calcular el tamaño y el escalado al lienzo de dibujado por si la pantalla se ha redimensionado
    const scaleX = overlayCanvas.width / Math.max(lastCaptureWidth, 1);
    const scaleY = overlayCanvas.height / Math.max(lastCaptureHeight, 1);
    const x = box.xmin * scaleX;
    const y = box.ymin * scaleY;
    const width = (box.xmax - box.xmin) * scaleX;
    const height = (box.ymax - box.ymin) * scaleY;

    // Pintar recuadro del objeto propiamente
    overlayContext.strokeStyle = color;
    overlayContext.lineWidth = 3;
    overlayContext.strokeRect(x, y, width, height);

    // Crear la etiqueta de texto con puntuación y borde redondeado (Apariencia de Píldora / Pill)
    const tag = `${label} · ${(score * 100).toFixed(0)}%`;
    overlayContext.font = "700 14px IBM Plex Sans, sans-serif";
    const textWidth = overlayContext.measureText(tag).width;
    const pillWidth = textWidth + 18;
    const pillHeight = 28;
    const pillX = Math.max(0, Math.min(x, overlayCanvas.width - pillWidth));
    const pillY = Math.max(0, y - pillHeight - 6);

    overlayContext.fillStyle = color;
    roundRect(overlayContext, pillX, pillY, pillWidth, pillHeight, 14);
    overlayContext.fill();

    // Dibujar el texto blanco dentro del bloque de relleno de color
    overlayContext.fillStyle = "#ffffff";
    overlayContext.fillText(tag, pillX + 9, pillY + 18);
  });
}

// ==========================================
// FUNCIONES DE APOYO / UTILIDADES
// ==========================================

/** Ajusta las dimensiones físicas del canvas al tamaño del vídeo proyectado en la pantalla. */
function syncCanvasSize() {
  const { clientWidth, clientHeight } = video;
  if (!clientWidth || !clientHeight) return;

  if (overlayCanvas.width !== clientWidth || overlayCanvas.height !== clientHeight) {
    overlayCanvas.width = clientWidth;
    overlayCanvas.height = clientHeight;
  }
}

/** Borra todos los dibujos previos. Útil por si el objeto detectado desaparece o borramos el área. */
function clearOverlay() {
  overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

/** Interrumpe el temporizador del ciclo de detección de imágenes. */
function stopDetectionLoop() {
  if (inferLoopTimer) {
    clearInterval(inferLoopTimer);
    inferLoopTimer = null;
  }
}

/** Desenchufa el streaming de cámara por completo (apaga la luz física indicadora si la tiene). */
function stopStream() {
  if (!mediaStream) return;

  mediaStream.getTracks().forEach((track) => track.stop());
  mediaStream = null;
  video.srcObject = null;
}

/** Atrapa y diagnostica excepciones a la hora de pedir permisos de cámara y muestra mensajes intuitivos a usuarios. */
function handleRuntimeError(error) {
  console.error(error);

  if (!navigator.mediaDevices?.getUserMedia) {
    statusLine.textContent = "Este navegador no soporta acceso a webcam desde esta página.";
    runtimeState.textContent = "No compatible";
    return;
  }

  if (error?.name === "NotAllowedError") {
    statusLine.textContent = "Permiso de cámara denegado. Permite el acceso y vuelve a intentarlo.";
    runtimeState.textContent = "Permiso denegado";
    return;
  }

  if (error?.name === "NotFoundError") {
    statusLine.textContent = "No se ha encontrado ninguna webcam disponible.";
    runtimeState.textContent = "Sin cámara";
    return;
  }

  statusLine.textContent = "No se pudo inicializar la demo. Revisa consola y conexión a internet.";
  runtimeState.textContent = "Error";
}

/** Función auxiliar matemática para dibujar esquinas redondeadas en el Canvas de HTML5 puro. */
function roundRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.lineTo(x + width - radius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + radius);
  context.lineTo(x + width, y + height - radius);
  context.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  context.lineTo(x + radius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - radius);
  context.lineTo(x, y + radius);
  context.quadraticCurveTo(x, y, x + radius, y);
  context.closePath();
}
