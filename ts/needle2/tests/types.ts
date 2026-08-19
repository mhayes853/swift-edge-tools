import type { Needle2Initialization, Needle2Runtime } from "@edge-tools/needle2";

// Type-level coverage for the inference `generate` performs over its tool list.
// Nothing here runs; `npm run test:types` fails if any assertion stops holding.

declare const runtime: Needle2Runtime;

export async function infersToolOutputsPerTool(): Promise<void> {
	const result = await runtime.generate({
		prompt: "weather in Paris, then email blob@gmail.com about it",
		initialization: {
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
		},
	});

	if (!result.success) {
		// @ts-expect-error a failed generation never invokes handlers
		result.functionCalls[0]?.output;
		return;
	}

	for (const call of result.functionCalls) {
		if (call.name === "get_weather") {
			const celsius: number = call.output.celsius;
			const city: string = call.arguments.city;
			void celsius;
			void city;
		} else if (call.name === "send_email") {
			const sent: string = call.output;
			void sent;
		} else {
			// @ts-expect-error a tool declared without `call` produces no output
			call.output;
		}
	}
}

export async function keepsUntypedToolListsWorking(): Promise<void> {
	const initialization: Needle2Initialization = {
		tools: [{ name: "set_thermostat", parameters: { type: "object" } }],
	};
	const result = await runtime.generate({ prompt: "21 degrees", initialization });
	const name: string = result.functionCalls[0]?.name ?? "";
	void name;
}

export async function keepsLoopOutputsInToolResponses(): Promise<void> {
	const response = await runtime.runLoop({
		prompt: "weather in Paris, then email blob@gmail.com about it",
		initialization: {
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
		},
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

export async function omitsOutputsFromRawGenerations(): Promise<void> {
	const generation = await runtime.generateRaw({
		prompt: "weather in Paris",
		initialization: {
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
		},
	});

	if (!generation.success) {
		return;
	}
	for (const call of generation.functionCalls) {
		const city: string = call.arguments.city;
		void city;
		// @ts-expect-error generateRaw never invokes handlers, so there is no output
		call.output;
	}
}
