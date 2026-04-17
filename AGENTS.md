# Repository Guidelines

## Project Structure & Module Organization
This repository is a small browser-only object detection demo. Keep UI structure in `index.html`, interaction and inference logic in `app.js`, and presentation rules in `style.css`. Store static assets in `img/` and keep model files under `models/onnx-community/rfdetr_nano-ONNX/` so the app can run offline without fetching weights from external services.

## Build, Test, and Development Commands
Install-free local development is the default.

- `npm start`: serves the project at `http://localhost:4173` using Python’s static server.
- `python3 -m http.server 8000`: simple fallback if you do not want to use the npm script.

Open the app in a modern browser with camera permissions enabled. Prefer Chromium-based browsers when checking WebGPU fallback behavior.

## Coding Style & Naming Conventions
Follow the existing style in the repo:

- Use 2-space indentation in HTML, CSS, and JavaScript.
- Use `const`/`let`, ES modules, and descriptive camelCase names such as `startCamera` or `currentThreshold`.
- Keep DOM ids and CSS classes kebab-case, for example `start-button` and `viewer-card`.
- Keep comments short and functional; explain why a fallback or browser constraint exists, not obvious syntax.

There is no formatter configured yet, so match surrounding code carefully and keep diffs small.

## Testing Guidelines
There is no automated test suite in the current project. Validate changes manually by:

- starting the local server,
- opening the app in the browser,
- confirming the model loads from `./models/`,
- verifying camera start/stop, threshold updates, and overlay rendering,
- checking that the app still falls back cleanly from WebGPU to WASM.

If you add automated checks later, place them in a top-level `tests/` folder and document the command in `package.json`.

## Commit & Pull Request Guidelines
Git history currently uses short, lowercase commit messages (`first commit`). Keep commits brief, imperative, and scoped, for example `adjust webcam status copy` or `fix wasm fallback handling`.

Pull requests should include a clear summary, manual test notes, and screenshots or short recordings for UI changes. Link the related issue when applicable and note any browser-specific behavior or model-file changes.

## Security & Configuration Tips
Do not point the app back to remote model hosting unless the change is intentional. Camera access, local model loading, and static hosting are core assumptions of this repository.
