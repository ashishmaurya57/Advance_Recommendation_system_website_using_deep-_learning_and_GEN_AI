import tailwindcss from "@tailwindcss/vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const backend = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [tanstackRouter({ target: "react", autoCodeSplitting: true }), react(), tailwindcss()],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  server: {
    port: 5173,
    // The API and the admin panel live on the FastAPI server.
    proxy: Object.fromEntries(
      // Keep the browser's Host header so the admin panel builds links for this origin.
      ["/api", "/admin"].map((path) => [path, { target: backend, changeOrigin: false }]),
    ),
  },
});
