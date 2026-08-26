import { copyFile, mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";

export default defineConfig(({ mode }) => {
  const entry =
    mode === "worker"
      ? "src/worker-entry.ts"
      : mode === "standalone"
        ? "src/standalone.ts"
        : mode === "zod"
          ? "src/zod.ts"
          : "src/index.ts";
  const fileName =
    mode === "worker"
      ? "needle2.worker.mjs"
      : mode === "standalone"
        ? "needle2.min.js"
        : mode === "zod"
          ? "zod.js"
          : "index.js";

  return {
    build: {
      emptyOutDir: mode === "production",
      lib: {
        entry: resolve(import.meta.dirname, entry),
        formats: [mode === "standalone" ? "iife" : "es"],
        name: mode === "standalone" ? "needle2" : undefined,
        fileName: () => fileName
      },
      minify: mode === "standalone",
      rollupOptions: {
        external: (id: string) => id.startsWith("node:") || id === "zod"
      },
      target: "es2022"
    },
    plugins: [
      rewriteAssetURLs(),
      ...(mode === "production" ? [copyAssets()] : [])
    ]
  };
});

function rewriteAssetURLs(): Plugin {
  const replacements = {
    __needle2_wasm__: "./needle.wasm",
    __needle2_weights__: "./needle2.cact",
    __needle2_worker__: "./needle2.worker.mjs"
  };
  return {
    name: "rewrite-needle2-asset-urls",
    renderChunk(code) {
      const rewritten = Object.entries(replacements).reduce(
        (result, [marker, path]) => result.replaceAll(marker, path),
        code
      );
      return rewritten.replace(
        /new URL\(\s*\/\* @vite-ignore \*\/\s*("\.\/needle(?:2\.cact|2\.worker\.mjs|\.wasm)")\s*,\s*import\.meta\.url\s*\)/g,
        "new URL($1, import.meta.url)"
      );
    }
  };
}

function copyAssets(): Plugin {
  return {
    name: "copy-needle2-assets",
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
