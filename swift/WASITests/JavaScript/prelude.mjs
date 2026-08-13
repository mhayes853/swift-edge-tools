import { readFile } from "node:fs/promises";
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

export async function setupOptions(options) {
	return options;
}
