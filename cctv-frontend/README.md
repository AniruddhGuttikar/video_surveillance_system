## Development — running the frontend

This project uses pnpm + Vite for the frontend. The steps below will install dependencies and start a development server with hot module reloading.

1. Install dependencies (uses pnpm):

```bash
pnpm install
```

2. Start the dev server:

```bash
pnpm run dev
```

Notes:

- The dev server runs via Vite and typically serves on http://localhost:5173 (the exact port will be printed when you run `pnpm run dev`).
- If you don't have pnpm installed, install it with npm: `npm install -g pnpm` or use your preferred package manager.
- For production builds use `pnpm run build` and then `pnpm run preview` to locally preview the build.

Troubleshooting:

- If port conflicts occur, pass a different port to Vite: `pnpm run dev -- --port 3000`.
- If you run into permission issues installing global packages, prefer a node version manager (nvm, asdf) or use `corepack` to enable pnpm bundled with Node.js.
