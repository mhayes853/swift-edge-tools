import { z } from "zod";
import type { Needle2JSONObject } from "./value.js";
import type { Needle2ToolOutput } from "./runtime.js";

export function zodTool<
	Name extends string,
	Schema extends z.ZodObject<z.core.$ZodShape>,
	Output extends Needle2ToolOutput = Needle2ToolOutput,
>(tool: {
	name: Name;
	description?: string;
	parameters: Schema;
	call: (args: z.infer<Schema>) => Output | PromiseLike<Output>;
}): {
	name: Name;
	description?: string;
	parameters: Needle2JSONObject;
	call: (args: z.input<Schema>) => Output | PromiseLike<Output>;
} {
	return {
		name: tool.name,
		...(tool.description === undefined ? {} : { description: tool.description }),
		parameters: zodToolParameters(tool.parameters),
		call: (args) => tool.call(tool.parameters.parse(args)),
	};
}

export function zodToolParameters<Schema extends z.ZodObject<z.core.$ZodShape>>(
	parameters: Schema,
): Needle2JSONObject {
	return z.toJSONSchema(parameters) as Needle2JSONObject;
}
