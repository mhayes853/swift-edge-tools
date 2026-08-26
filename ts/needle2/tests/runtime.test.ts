import { afterEach, describe, expect, test } from "vitest";
import { needle2 } from "@edge-tools/needle2";
import type {
	Needle2Factory,
	Needle2Provider,
	Needle2Runtime,
} from "@edge-tools/needle2";
import { thermostatRequest, thermostatTools } from "./support.ts";

const runtimes: Needle2Runtime<any>[] = [];
const emailTools = [
		{
			name: "send_email",
			description: "Sends an email to a recipient with an email address.",
			parameters: {
				type: "object",
				properties: {
					address: {
						type: "string",
						pattern: "[a-z][a-z0-9]{1,10}@gmail\\.com",
						description: "The recipient's email address.",
						examples: ["blob@gmail.com"],
					},
					subject: {
						type: "string",
						description: "The subject of an email.",
					},
					body: {
						type: "string",
						description: "The content of an email.",
					},
				},
				required: ["address", "subject", "body"],
			},
		},
	] as const;
const weatherTools = [
		{
			name: "get_weather",
			description: "Get the weather for a city.",
			parameters: {
				type: "object",
				properties: { city: { type: "string" } },
				required: ["city"],
			},
		},
	] as const;

afterEach(async () => {
	await Promise.allSettled(runtimes.splice(0).map((runtime) => runtime.dispose()));
});

test("returns the native error response for truncated generation", async () => {
	const runtime = await needle2({ provider: "direct", tools: emailTools });
	runtimes.push(runtime);

	const result = await runtime.generate({
		prompt: "Send an email to blob@gmail.com asking them to go hiking.",
		maxTokens: 4,
  });

	expect(result.success).toBe(false);
	expect(result.type).toBe("call");
	expect(result.tokenCount).toBe(4);
	if (result.success) {
		throw new Error("Expected Needle 2 generation to be truncated.");
	}
	expect(result.error).toBe("tool call truncated: token budget exhausted");
	expect(result.errorCode).toBe("truncated");
});

const weatherTool = {
	name: "get_weather",
	description: "Get the current weather for a city.",
	parameters: {
		type: "object",
		properties: { city: { type: "string" } },
		required: ["city"],
	},
} as const;

const emailTool = {
	name: "send_email",
	description: "Sends an email to a recipient with an email address.",
	parameters: {
		type: "object",
		properties: {
			address: { type: "string" },
			subject: { type: "string" },
			body: { type: "string" },
		},
		required: ["address", "subject", "body"],
	},
} as const;

const weatherAndEmailPrompt =
	"What's the weather in Paris, and email blob@gmail.com about it?";
const toolCallResponse = JSON.stringify({
	success: true,
	type: "call",
	function_calls: [{ name: "pause", arguments: {} }],
});
const textResponse = JSON.stringify({ success: true, type: "respond" });

test("rejects duplicate tool names", async () => {
	await expect(
		needle2({
			provider: "direct",
			tools: [weatherTool, weatherTool],
		}),
	).rejects.toMatchObject({ code: "duplicate-tool-name" });
});

test("generate leaves tool execution to the caller", async () => {
	let invocations = 0;
	const runtime = await needle2({
		provider: "direct",
		tools: [
			{
				...weatherTool,
				call: () => {
					invocations += 1;
					return "sunny";
				},
			},
			{ ...emailTool, call: () => "sent" },
		],
	});
	runtimes.push(runtime);

	const generation = await runtime.generate({
		prompt: weatherAndEmailPrompt,
	});

	expect(generation.success).toBe(true);
	expect(generation.functionCalls.map((call) => call.name)).toEqual([
		"get_weather",
		"send_email",
	]);
	expect(generation.functionCalls[0]).not.toHaveProperty("output");
	expect(invocations).toBe(0);
});

test("drives the tool loop until the model responds", async () => {
	const runtime = await needle2({
		provider: "direct",
		tools: [
			{ ...weatherTool, call: () => ({ celsius: 21, condition: "sunny" }) },
			{ ...emailTool, call: () => ({ status: "sent" }) },
		],
	});
	runtimes.push(runtime);

	const response = await runtime.runLoop({
		prompt: weatherAndEmailPrompt,
	});

	expect(response.terminationCause).toBe("responded");
	expect(response.steps).toHaveLength(2);
	expect(response.steps[0]?.toolResponses).toEqual([
		{ celsius: 21, condition: "sunny" },
		{ status: "sent" },
	]);
	expect(response.steps[1]?.generation.functionCalls).toEqual([]);
	expect(response.steps[1]?.toolResponses).toEqual([]);
	expect(response.response).toBe(response.steps[1]?.generation);
});

test("feeds a failed tool handler back to the model instead of throwing", async () => {
	const runtime = await needle2({
		provider: "direct",
		tools: [
			{
				...weatherTool,
				call: () => {
					throw new Error("the weather station is down");
				},
			},
			{ ...emailTool, call: () => ({ status: "sent" }) },
		],
	});
	runtimes.push(runtime);

	const response = await runtime.runLoop({
		prompt: weatherAndEmailPrompt,
	});

	expect(response.steps[0]?.toolResponses).toEqual([
		{ error: "the weather station is down" },
		{ status: "sent" },
	]);
	expect(response.steps.length).toBeGreaterThan(1);
});

test("feeds an unknown tool error back for a tool declared without a handler", async () => {
	const runtime = await needle2({
		provider: "direct",
		tools: [{ ...weatherTool, call: () => ({ celsius: 21 }) }, emailTool],
	});
	runtimes.push(runtime);

	const response = await runtime.runLoop({
		prompt: weatherAndEmailPrompt,
	});

	expect(response.steps[0]?.toolResponses).toEqual([
		{ celsius: 21 },
		{ error: "unknown tool: send_email" },
	]);
});

test("stops the loop once it runs out of turns", async () => {
	const runtime = await needle2({
		provider: "direct",
		tools: [
			{ ...weatherTool, call: () => ({ celsius: 21 }) },
			{ ...emailTool, call: () => ({ status: "sent" }) },
		],
	});
	runtimes.push(runtime);

	const response = await runtime.runLoop({
		prompt: weatherAndEmailPrompt,
		maxTurns: 1,
	});

	expect(response.terminationCause).toBe("maximum-turns-reached");
	expect(response.steps).toHaveLength(1);
});

test("rejects a turn count Needle 2 cannot support", async () => {
	const runtime = await needle2({
		provider: "direct",
		tools: [{ ...weatherTool, call: () => ({ celsius: 21 }) }],
	});
	runtimes.push(runtime);

	await expect(
		runtime.runLoop({
			prompt: weatherAndEmailPrompt,
			maxTurns: 9,
		}),
	).rejects.toMatchObject({ code: "invalid-turns" });
});

test("serializes direct runtimes with different initializations", async () => {
	const thermostat = await needle2({ provider: "direct", tools: thermostatTools });
	const weather = await needle2({ provider: "direct", tools: weatherTools });
	runtimes.push(thermostat, weather);

	const thermostatPrompt = {
		prompt: "set it to 21 degrees",
	};
	const weatherPrompt = {
		prompt: "what's the weather in Paris?",
	};

	const settled = await Promise.allSettled([
		thermostat.generate(thermostatPrompt),
		weather.generate(weatherPrompt),
	]);
	const [thermostatResult, weatherResult] = settled.map((result) => {
		if (result.status === "rejected") {
			throw result.reason;
		}
		return result.value;
	});
	expect(thermostatResult.functionCalls[0]?.name).toBe("set_thermostat");
	expect(weatherResult.functionCalls[0]?.name).toBe("get_weather");
	expect(
		(await thermostat.generate(thermostatPrompt)).functionCalls[0]?.name,
	).toBe("set_thermostat");
});

test("keeps a direct model leased while a tool loop is suspended", async () => {
	const prompts: string[] = [];
	const factory = scriptedFactory((prompt) => {
		prompts.push(prompt);
		return prompt === "loop" ? toolCallResponse : textResponse;
	});
	const paused = await pausingRuntime(factory);
	const second = await directRuntime(factory);
	runtimes.push(paused.runtime, second);

	const loop = paused.runtime.runLoop({ prompt: "loop" });
	await paused.started;
	const generation = second.generate({ prompt: "other" });
	await new Promise((resolve) => setTimeout(resolve, 0));
	const promptsWhileSuspended = [...prompts];
	paused.release();
	const completed = await Promise.allSettled([loop, generation]);

	expect(promptsWhileSuspended).toEqual(["loop"]);
	expect(completed.every((result) => result.status === "fulfilled")).toBe(true);
});

describe.each([
	"direct",
	"worker",
] satisfies Needle2Provider[])("Needle2Runtime with the %s provider", (provider) => {
	test("generates a parsed response", async () => {
		const runtime = await needle2({ provider, tools: emailTools });
		runtimes.push(runtime);

		const result = await runtime.generate({
			prompt: "Send an email to blob@gmail.com asking them to go hiking.",
		});

		expect(result.success).toBe(true);
		expect(result.type).toBe("call");
		expect(result.tokenCount).toBeGreaterThan(0);
		expect(result.functionCalls).toHaveLength(1);
		expect(result.functionCalls[0]).toMatchObject({
			name: "send_email",
			arguments: { address: "blob@gmail.com" },
		});
		expect(result.metrics.peakRAMMegabytes).toBeUndefined();
		expect(result).toMatchSnapshot({
			metrics: {
				prefillTokensPerSecond: expect.any(Number),
				decodeTokensPerSecond: expect.any(Number),
			},
		});
	});

	test("runs tool handlers across the provider boundary", async () => {
		const runtime = await needle2({
			provider,
			tools: [
				{
					name: "set_thermostat",
					description: "Set the thermostat temperature.",
					parameters: {
						type: "object",
						properties: { temperature: { type: "integer" } },
						required: ["temperature"],
					},
					call: async (args: { temperature: number }) => ({
						status: "ok" as const,
						temperature: args.temperature,
					}),
				},
			],
		});
		runtimes.push(runtime);

		const result = await runtime.runLoop({
			prompt: "set the thermostat to 21 degrees",
			maxTurns: 1,
		});

		expect(result.steps[0]?.toolResponses[0]).toEqual({
			status: "ok",
			temperature: 21,
		});
	});

	test("rejects generation after disposal", async () => {
		const runtime = await needle2({ provider, tools: thermostatTools });
		runtimes.push(runtime);
		await runtime.dispose();

		await expect(runtime.generate(thermostatRequest)).rejects.toMatchObject({
			code: "disposed",
		});
	});
});

test("shares initial weights but reloads an explicitly loaded source", async () => {
	let loadCount = 0;
	const factory: Needle2Factory = () => ({
		HEAPU8: new Uint8Array(16),
		_needle_load: () => {
			loadCount += 1;
			return 0;
		},
		_needle_reset() {},
		_malloc: () => 1,
		_free() {},
		UTF8ToString: () => "",
		ccall: () => 0,
	});
	const wasm = new Uint8Array([0]);
	const weights = new Uint8Array([1]);
	const first = await needle2({
		provider: "direct",
		factory,
		wasm,
		weights,
	});
	const second = await needle2({
		provider: "direct",
		factory,
		wasm,
		weights,
	});
	runtimes.push(first, second);

	expect(loadCount).toBe(1);
	await first.load(weights);
	expect(loadCount).toBe(2);
});

test("initializes a direct runtime again after reset", async () => {
	let initializationCount = 0;
	const factory: Needle2Factory = () => ({
		HEAPU8: new Uint8Array(65_537),
		_needle_load: () => 0,
		_needle_reset() {},
		_malloc: () => 1,
		_free() {},
		UTF8ToString: () => '{"success":true,"type":"respond"}',
		ccall: (name: string) => {
			if (name === "needle_init") {
				initializationCount += 1;
			}
			return name === "needle_complete" ? 1 : 0;
		},
	});
	const runtime = await needle2({
		provider: "direct",
		factory,
		wasm: new Uint8Array([0]),
		weights: new Uint8Array([1]),
	});
	runtimes.push(runtime);

	await runtime.generate({ prompt: "first" });
	await runtime.generate({ prompt: "second" });
	expect(initializationCount).toBe(1);

	await runtime.reset();
	await runtime.generate({ prompt: "third" });
	expect(initializationCount).toBe(2);
});

describe.each([
	"direct",
	"worker",
] satisfies Needle2Provider[])("Needle2Runtime auto engine with the %s provider", (provider) => {
	test("falls back to WASM when native artifacts are unavailable", async () => {
		const runtime = await needle2({ provider, engine: "auto", tools: thermostatTools });
		runtimes.push(runtime);

		const result = await runtime.generate(thermostatRequest);

		expect(result.success).toBe(true);
		expect(result.functionCalls[0]?.name).toBe("set_thermostat");
	});
});

function scriptedFactory(
	response: (prompt: string) => string,
): Needle2Factory {
	let output = textResponse;
	return () => ({
		HEAPU8: new Uint8Array(65_537),
		_needle_load: () => 0,
		_needle_reset() {},
		_malloc: () => 1,
		_free() {},
		UTF8ToString: () => output,
		ccall: (name: string, ...args: unknown[]) => {
			if (name === "needle_complete") {
				const argumentValues = args[2] as readonly unknown[];
				output = response(String(argumentValues[0]));
				return 1;
			}
			return 0;
		},
	});
}

async function pausingRuntime(factory: Needle2Factory) {
	let toolStarted!: () => void;
	const started = new Promise<void>((resolve) => {
		toolStarted = resolve;
	});
	let release!: () => void;
	const released = new Promise<void>((resolve) => {
		release = resolve;
	});
	const runtime = await needle2({
		provider: "direct",
		factory,
		wasm: new Uint8Array([0]),
		weights: new Uint8Array([1]),
		tools: [
			{
				name: "pause",
				parameters: { type: "object", properties: {} },
				call: async () => {
					toolStarted();
					await released;
					return "done";
				},
			},
		],
	});
	return { runtime, started, release };
}

function directRuntime(factory: Needle2Factory) {
	return needle2({
		provider: "direct",
		factory,
		wasm: new Uint8Array([0]),
		weights: new Uint8Array([1]),
	});
}
