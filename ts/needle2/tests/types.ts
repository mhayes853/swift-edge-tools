import { needle2 } from "@edge-tools/needle2";
import type { Needle2Runtime } from "@edge-tools/needle2";

// Type-level coverage for the inference performed over each tool list.
// Nothing here runs; `npm run test:types` fails if any assertion stops holding.

export async function infersToolArgumentsPerTool(): Promise<void> {
	const runtime = await needle2({
		provider: "direct",
		tools: [
				{
					name: "get_weather",
					description: "Get the current weather for a city.",
					parameters: {
						type: "object",
						properties: { city: { type: "string" } },
						required: ["city"],
					},
					call: async (args: { city: string }) => ({ celsius: 21, city: args.city }),
				},
				{
					name: "send_email",
					parameters: {
						type: "object",
						properties: { address: { type: "string" } },
						required: ["address"],
						examples: ["blob@gmail.com"],
					},
					call: (args: { address: string }) => `sent to ${args.address}`,
				},
				{
					name: "log_event",
					parameters: { type: "object", properties: {} },
				},
		],
	});
	const result = await runtime.generate({
		prompt: "weather in Paris, then email blob@gmail.com about it",
	});

	for (const call of result.functionCalls) {
		if (call.name === "get_weather") {
			const city: string = call.arguments.city;
			void city;
		} else if (call.name === "send_email") {
			const address: string = call.arguments.address;
			void address;
		}
		// @ts-expect-error generate never invokes handlers
		call.output;
	}
}

export async function keepsUntypedToolListsWorking(): Promise<void> {
	const runtime: Needle2Runtime = await needle2({
		provider: "direct",
		tools: [{ name: "set_thermostat", parameters: { type: "object" } }],
	});
	const result = await runtime.generate({ prompt: "21 degrees" });
	const name: string = result.functionCalls[0]?.name ?? "";
	void name;
}

export async function keepsLoopOutputsInToolResponses(): Promise<void> {
	const runtime = await needle2({
		provider: "direct",
		tools: [
				{
					name: "get_weather",
					parameters: {
						type: "object",
						properties: { city: { type: "string" } },
						required: ["city"],
					},
					call: async (args: { city: string }) => ({ celsius: 21, city: args.city }),
				},
		],
	});
	const response = await runtime.runLoop({
		prompt: "weather in Paris, then email blob@gmail.com about it",
	});

	const cause: "responded" | "refused" | "no-tool-calls" | "maximum-turns-reached" | "failed" =
		response.terminationCause;
	void cause;

	for (const step of response.steps) {
		const responses: unknown[] = step.toolResponses;
		void responses;
		if (!step.generation.success) {
			continue;
		}
		for (const call of step.generation.functionCalls) {
			const city: string = call.arguments.city;
			void city;
			// @ts-expect-error loop outputs live in `toolResponses`, not on the call
			call.output;
		}
	}
}
