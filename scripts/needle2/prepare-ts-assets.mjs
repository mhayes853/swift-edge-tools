import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const revision = "17a803d95928ba33d3e9a0160e024d9565b5c3f2";
const repository = `https://huggingface.co/Cactus-Compute/needle2/resolve/${revision}`;
const checksums = {
  "wasm/needle.js":
    "06499ec635d7e2790cb84791bc0e323fa4d0c5a8948108ca357b76685e085a66",
  "wasm/needle.wasm":
    "4bd9538633573078419ba09b8d2e943d7a99abcdb67bf3c7dd874055e1e10fc8",
  "needle2.cact":
    "b43aabfcaf1a6db6acf488076eab71d823c08697c7af4521fc1d174b60ede5ba",
  LICENSE: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
};

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryDirectory = resolve(scriptDirectory, "../..");
const packageDirectory = resolve(repositoryDirectory, "ts/needle2");

const [binding, wasm, weights, license] = await Promise.all([
  download("wasm/needle.js"),
  download("wasm/needle.wasm"),
  download("needle2.cact"),
  download("LICENSE"),
]);

await mkdir(resolve(packageDirectory, "assets"), { recursive: true });
await mkdir(resolve(packageDirectory, "vendor"), { recursive: true });
await writeFile(resolve(packageDirectory, "LICENSE-Needle2"), license);
await writeFile(resolve(packageDirectory, "assets/needle.wasm"), wasm);
await writeFile(resolve(packageDirectory, "assets/needle2.cact"), weights);
const bindingSource = new TextDecoder()
  .decode(binding)
  .replace(
    'var fs=require("node:fs");scriptDirectory=__dirname+"/";',
    'var fs=await import("node:fs");var nodeCrypto=await import("node:crypto");scriptDirectory=new URL(/* @vite-ignore */".",import.meta.url).pathname;',
  )
  .replace('var nodeCrypto=require("node:crypto");return', "return");
if (bindingSource.includes('require("node:')) {
  throw new Error(
    "The normalized Needle binding still contains Node.js require calls.",
  );
}
await writeFile(
  resolve(packageDirectory, "vendor/needle.mjs"),
  `/* Modified by Swift Edge Tools to support ESM Node imports and a default export. */\n${bindingSource}\nexport default createNeedle;\n`,
);

console.log(
  `Prepared Needle 2 JavaScript, WASM, and default weights at revision ${revision}.`,
);

async function download(path) {
  const response = await fetch(`${repository}/${path}`);
  if (!response.ok) {
    throw new Error(`Failed to download ${path}: HTTP ${response.status}.`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  const checksum = createHash("sha256").update(bytes).digest("hex");
  if (checksum !== checksums[path]) {
    throw new Error(`Unexpected checksum for ${path}: ${checksum}.`);
  }
  return bytes;
}
