# Webcam Detector (Browser Web)

Este proyecto docente muestra cómo crear una aplicación web ligera de **Visión Artificial y Detección de Objetos en tiempo real**, capaz de ejecutarse nativamente en el navegador **sin necesidad de backend** ni servidores remotos de inferencia.

Es el punto de entrada ideal para ver la potencia de la IA moderna integrada en la web, de forma educativa.

## Características principales

* **Privacidad y procesamiento local:** Usa la librería [Transformers.js](https://huggingface.co/docs/transformers.js/) de Hugging Face. El modelo (modelo cuantizado `rfdetr_nano-ONNX`) se almacena localmente y corre dentro del navegador web local del usuario. Las imágenes de la cámara nunca se envían a internet; todo el proceso es privado.
* **Aceleración de Hardware Dinámica:** Implementa un mecanismo de "fallback" (rescate). Intenta utilizar primero tu tarjeta gráfica (*GPU*) vía API originaria de **WebGPU** para maximizar rendimiento y subir fotogramas por segundo (FPS). Si el navegador o hardware no lo soporta, cae ordenadamente a la predicción tradicional usando el procesador (*CPU*) mediante **WASM (WebAssembly)**.
* **Auto-alojado (Self-hosted para GitHub Pages):** Está configurado de serie para no depender de la pasarela de internet externa. Los pesos matemáticos del modelo pre-entrenado se almacenan en la carpeta `/models`. Esto permite servir la app universalmente desde cualquier plataforma de archivos estáticos (incluyendo **GitHub Pages**).
* **Integración WebMedia e UI Dinámica:** Recoge secuencias de cámara (*streams*) usando `navigator.mediaDevices.getUserMedia` transparente y de alto rendimiento, e inyecta dibujos (cajas delimitadoras / bounding boxes) superpuestos mediante la interfaz pura de *Canvas HTML5*, actualizándose repetidamente mediante temporizadores nativos en JavaScript.

## Estructura del Proyecto

* `index.html`: La estructura visual e interfaz gráfica para el usuario.
* `style.css`: Hojas de estilo para un acabado limpio y moderno.
* `app.js`: El cerebro de la demo. Un archivo ampliamente comentado, con flujos fáciles de entender en torno a la cámara y a las matemáticas del detector.
* `models/`: Carpeta que acoge el peso matemático offline del modelo `onnx-community/rfdetr_nano-ONNX`, tanto para CPU como para GPU WebGL/WebGPU.

## Cómo ejecutar el proyecto en tu PC

Puesto que la aplicación utiliza APIs críticas del navegador vinculadas a la seguridad y la privacidad (como el `getUserMedia` para activar micrófonos/cámaras, o directivas de importación de módulos CORS y WebGPU), los navegadores bloquean la apertura por doble clic usando el protocolo local "file:///".

Por esto, **necesitas servir la carpeta de desarrollo usando un servidor HTTP estático básico.**

### Alternativa 1: La más fácil (VS Code)
1. Abre esta carpeta (`projects/browser-web/solucion`) en Visual Studio Code.
2. Descarga e instala la extensión **"Live Server"** desde el panel lateral de Extensiones (busca el desarrollador Ritwick Dey).
3. Aparecerá en la barra azul, abajo a la derecha, un botón que dice `Go Live`. Púlsalo. ¡Listo! Se abrirá Chrome/Edge en tu aplicación.

### Alternativa 2: Desde terminal con Python
Cualquier ordenador con Python 3 instalado, incluye ya un servidor web de una línea.
Abre tu consola de comandos en esta misma carpeta y ejecuta:
```bash
python -m http.server 8000
```
Entra en tu navegador a: `http://localhost:8000`

### Alternativa 3: Puesta en Producción / Cloud
Gracias a la propiedad "Auto-alojada" (*self-hosting*) implementada en `app.js` desactivando los llamados externos a Hugging Face y sirviendo desde `./models/`:
¡Puedes directamente encender **GitHub Pages** en la rama principal de tu repositorio! Una vez se completen las acciones automáticas (Actions) de Git, tendrás una URL pública para enseñársela a tus alumnos en toda la clase.
