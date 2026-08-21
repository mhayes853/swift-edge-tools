import { describe, expect, test } from "vitest";
import { z } from "zod";
import { zodTool, zodToolParameters } from "@edge-tools/needle2/zod";

describe("Needle2 zod tool tests", () => {
	test("derives the JSON schema from the zod object schema", () => {
		const tool = zodTool({
			name: "get_weather",
			description: "Get the weather for a city.",
			parameters: z.object({ city: z.string(), days: z.number().int().optional() }),
			call: () => null
		});

		expect(tool.parameters).toMatchObject({
			type: "object",
			properties: {
				city: { type: "string" },
				days: { type: "integer" }
			},
			required: ["city"]
		});
	});

	test("coerces and fills in defaults declared on the schema before calling the handler", () => {
		const tool = zodTool({
			name: "set_brightness",
			parameters: z.object({
				level: z.coerce.number(),
				unit: z.string().default("percent")
			}),
			call: (args) => args
		});

		expect(tool.call({ level: "42" })).toEqual({
			level: 42,
			unit: "percent"
		});
	});

	test("rejects arguments that don't satisfy the schema instead of calling the handler", () => {
		const tool = zodTool({
			name: "set_brightness",
			parameters: z.object({ level: z.number().max(100) }),
			call: () => {
				throw new Error("handler should not run");
			}
		});

		expect(() => tool.call({ level: 200 })).toThrow(z.ZodError);
	});

	test("zodToolParameters converts a schema without requiring a handler", () => {
		expect(zodToolParameters(z.object({ query: z.string() }))).toMatchObject({
			type: "object",
			properties: { query: { type: "string" } },
			required: ["query"]
		});
	});
});
