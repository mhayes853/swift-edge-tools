import { copyFile, mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";

export default defineConfig(({ mode }) => {
  const entry =
    mode === "worker"
      ? "src/worker-entry.ts"
      : mode === "standalone"
        ? "src/standalone.ts"
        : "src/index.ts";
  const fileName =
    mode === "worker"
      ? "needle2.worker.mjs"
      : mode === "standalone"
        ? "needle2.min.js"
        : "index.js";

  return {
    build: {
      emptyOutDir: mode === "production",
      lib: {
        entry: resolve(import.meta.dirname, entry),
        formats: [mode === "standalone" ? "iife" : "es"],
        name: mode === "standalone" ? "needle2Runtime" : undefined,
        fileName: () => fileName
      },
      minify: mode === "standalone",
      rollupOptions: {
        external: (id: string) => id.startsWith("node:")
      },
      target: "es2022"
    },
    plugins: [copyWASM()]
  };
});

function copyWASM(): Plugin {
  return {
    name: "copy-needle2-wasm",
    async writeBundle() {
      const outputDirectory = resolve(import.meta.dirname, "dist");
      await mkdir(outputDirectory, { recursive: true });
      await copyFile(
        resolve(import.meta.dirname, "assets/needle.wasm"),
        resolve(outputDirectory, "needle.wasm")
      );
      await copyFile(
        resolve(import.meta.dirname, "assets/needle2.cact"),
        resolve(outputDirectory, "needle2.cact")
      );
    }
  };
}
