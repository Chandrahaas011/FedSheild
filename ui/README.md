# Research Console Frontend

All frontend code, packages, and compiled assets are isolated in this subfolder.
Stack: React 19, TypeScript, Vite 6, Tailwind 4, Phosphor icons. No cloud services,
remote fonts, or Python web-framework dependencies are required.

```powershell
npm ci --cache ../.tools/npm-cache --no-audit --no-fund
npm run build
```

From the repository root, serve the complete built application:

```powershell
.\.venv\Scripts\python.exe -m src.demo.server
```

Open http://127.0.0.1:8765. For development, run `npm run dev` here while Python
remains running; Vite proxies `/api` to port 8765. The full setup and showcase script,
including the working npm fallback on this computer, are in
[UI_DEMONSTRATION_GUIDE.md](../docs/UI_DEMONSTRATION_GUIDE.md).

The home page separates saved results from fresh, guided in-memory recovery.
`components/Home.tsx` supplies a palette-matched, fluid-inspired CSS background;
`components/Lifecycle.tsx` drives eight presenter-paced backend stages. Conditional
resumption affects only session predictions, not source artifacts or deployed models.

The Python adapter is separate in `src/demo/`. It requires existing local processed
data and trusted saved artifacts. It never changes experiment evidence. Recovery
views show saved decisions; the separate guided page demonstrates actual in-memory
fitting and validation, not deployment activation. Builds and
`node_modules/` are ignored by Git; `package-lock.json` pins reproducible installs.
