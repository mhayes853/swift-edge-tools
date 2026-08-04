import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const packageName = process.env.EDGE_TOOLS_ONNX_PACKAGE ?? "onnxruntime-node";
const importedRuntime = await import(packageName);
const importedWebRuntime =
	packageName === "onnxruntime-web"
		? importedRuntime
		: await import("onnxruntime-web");

globalThis.edgeToolsONNXRuntime = importedRuntime;
globalThis.edgeToolsWebONNXRuntime = importedWebRuntime;
globalThis.edgeToolsRejectedONNXRuntime = {
	InferenceSession: {
		create() {
			return Promise.reject(new Error("The test session was rejected."));
		},
	},
	Tensor: class {},
};
globalThis.edgeToolsAddModel = new Uint8Array(
	await readFile(
		fileURLToPath(
			new URL("../Tests/EdgeToolsWASITests/Fixtures/add.onnx", import.meta.url),
		),
	),
);

const defaultNeedleDirectory = path.resolve(
	fileURLToPath(new URL("../../..", import.meta.url)),
	".edge-tools-tests/onnx-export-v2-float32-int4",
);
const needleDirectory =
	process.env.EDGE_TOOLS_NEEDLE_MODEL_DIRECTORY ?? defaultNeedleDirectory;
const needleFiles = [
	"encoder.onnx",
	"encoder.onnx.data",
	"decoder.onnx",
	"decoder.onnx.data",
	"tokenizer.model",
];

if (needleFiles.every((file) => existsSync(path.join(needleDirectory, file)))) {
	const contents = await Promise.all(
		needleFiles.map(async (file) => [
			file,
			new Uint8Array(await readFile(path.join(needleDirectory, file))),
		]),
	);
	globalThis.edgeToolsNeedleFixture = Object.fromEntries(contents);
}

export async function setupOptions(options) {
	return options;
}
