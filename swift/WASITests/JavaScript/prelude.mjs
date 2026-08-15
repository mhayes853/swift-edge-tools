import {
	defaultSystemValues,
	needle2,
} from "../../../ts/needle2/dist/index.js";

globalThis.edgeToolsNeedle2Runtime = needle2;

export async function setupOptions(options) {
	const getImports = options.getImports;
	return {
		...options,
		getImports(context) {
			return {
				...(getImports?.(context) ?? {}),
				needle2DefaultSystemValues: (options) => defaultSystemValues(options),
			};
		},
	};
}
