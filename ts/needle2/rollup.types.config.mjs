import { rm } from "node:fs/promises";
import { dts } from "rollup-plugin-dts";

const declarations = ["index", "zod", "standalone"];

export default declarations.map((name, index) => ({
	input: `dist/types/${name}.d.ts`,
	output: { file: `dist/${name}.d.ts`, format: "es" },
	plugins: [
		dts(),
		...(index === declarations.length - 1 ? [removeIntermediateTypes()] : []),
	],
}));

function removeIntermediateTypes() {
	return {
		name: "remove-intermediate-types",
		closeBundle() {
			return rm("dist/types", { recursive: true });
		},
	};
}
