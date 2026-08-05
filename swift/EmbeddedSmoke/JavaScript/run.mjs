import fs from "node:fs";
import { WASI } from "node:wasi";

const modulePath = process.argv[2];
if (!modulePath) throw new Error("Usage: run.mjs <module.wasm>");

const wasi = new WASI({ version: "preview1", args: ["smoke"], env: {}, returnOnExit: true });
const module = await WebAssembly.compile(fs.readFileSync(modulePath));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());

let code;
try {
	code = wasi.start(instance) ?? 0;
} catch (error) {
	console.error(`Trapped: ${error.message}`);
	process.exit(1);
}
process.exit(code);
