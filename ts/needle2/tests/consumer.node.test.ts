import { mkdtemp, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, expect, test } from "vitest";
import { build } from "vite";

const outputDirectories: string[] = [];

afterEach(async () => {
  await Promise.allSettled(
    outputDirectories.splice(0).map((directory) =>
      rm(directory, { recursive: true, force: true })
    )
  );
});

test("a consumer Vite build emits every runtime asset", async () => {
  const outputDirectory = await mkdtemp(join(tmpdir(), "needle2-consumer-"));
  outputDirectories.push(outputDirectory);

  await build({
    configFile: false,
    root: resolve(import.meta.dirname, "consumer"),
    logLevel: "silent",
    build: { outDir: outputDirectory }
  });

  const files = await recursiveFiles(outputDirectory);
  expect(files.some((file) => file.endsWith(".wasm"))).toBe(true);
  expect(files.some((file) => file.endsWith(".cact"))).toBe(true);
  expect(files.some((file) => file.includes("needle2.worker"))).toBe(true);
});

async function recursiveFiles(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const results = await Promise.allSettled(
    entries.map((entry) => {
      const path = join(directory, entry.name);
      return entry.isDirectory() ? recursiveFiles(path) : [path];
    })
  );
  return results.flatMap((result) => {
    if (result.status === "rejected") throw result.reason;
    return result.value;
  });
}
