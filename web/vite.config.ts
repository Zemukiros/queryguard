import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The API runs on :8000 (make api). Proxying keeps the browser same-origin in
// dev and preview, so CORS is only needed for a separately hosted frontend.
const api = process.env.QUERYGUARD_API ?? "http://127.0.0.1:8000";
const proxy = { "/v1": api, "/healthz": api };

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, strictPort: true, proxy },
  preview: { port: 4173, strictPort: true, proxy },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
