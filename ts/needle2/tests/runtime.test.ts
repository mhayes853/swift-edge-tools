import { afterEach, describe, expect, test } from "vitest";
import { needle2 } from "@edge-tools/needle2";
import type {
	Needle2Initialization,
	Needle2Factory,
	Needle2Provider,
	Needle2Runtime,
} from "@edge-tools/needle2";
import { thermostatInitialization, thermostatRequest } from "./support.ts";

const runtimes: Needle2Runtime[] = [];
const emailInitialization: Needle2Initialization = {
	tools: [
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
	],
};
const weatherInitialization: Needle2Initialization = {
	tools: [
		{
			name: "get_weather",
			description: "Get the weather for a city.",
			parameters: {
				type: "object",
				properties: { city: { type: "string" } },
				required: ["city"],
			},
		},
	],
};

afterEach(async () => {
	await Promise.all(runtimes.splice(0).map((runtime) => runtime.dispose()));
});

test("returns the native error response for truncated generation", async () => {
	const runtime = await needle2({ provider: "direct" });
	runtimes.push(runtime);

	const result = await runtime.generate({
		prompt: "Send an email to blob@gmail.com asking them to go hiking.",
		initialization: emailInitialization,
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

// `as const` keeps the literal tool names when these are spread into a tool with a
// handler, which is what drives the per-tool typing of `output`.
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

test("invokes tool handlers in parallel and attaches their outputs", async () => {
	const runtime = await needle2({ provider: "direct" });
	runtimes.push(runtime);

	// The first handler only settles once the second has started, so a sequential
	// implementation deadlocks rather than quietly passing.
	let startSecond!: () => void;
	const second = new Promise<void>((resolve) => {
		startSecond = resolve;
	});

	const result = await runtime.generate({
		prompt: weatherAndEmailPrompt,
		initialization: {
			tools: [
				{
					...weatherTool,
					call: async (args: { city: string }) => {
						await second;
						return { celsius: 21, city: args.city };
					},
				},
				{
					...emailTool,
					call: (args: { address: string }) => {
						startSecond();
						return `sent to ${args.address}`;
					},
				},
			],
		},
	});

	expect(result.success).toBe(true);
	if (!result.success) {
		throw new Error("Expected Needle 2 to call both tools.");
	}
	expect(result.functionCalls.map((call) => call.name)).toEqual([
		"get_weather",
		"send_email",
	]);
	expect(result.functionCalls[0]?.output).toEqual({
		celsius: 21,
		city: "Paris",
	});
	expect(result.functionCalls[1]?.output).toBe("sent to blob@gmail.com");
});

test("leaves tools declared without a handler uninvoked", async () => {
	const runtime = await needle2({ provider: "direct" });
	runtimes.push(runtime);

	const result = await runtime.generate({
		prompt: weatherAndEmailPrompt,
		initialization: {
			tools: [
				{ ...weatherTool, call: () => "sunny" },
				emailTool,
			],
		},
	});

	expect(result.success).toBe(true);
	if (!result.success) {
		throw new Error("Expected Needle 2 to call both tools.");
	}
	expect(result.functionCalls[0]).toMatchObject({
		name: "get_weather",
		output: "sunny",
	});
	expect(result.functionCalls[1]).not.toHaveProperty("output");
});

test("reports every failed tool handler without losing the generation", async () => {
	const runtime = await needle2({ provider: "direct" });
	runtimes.push(runtime);

	const failure = await runtime
		.generate({
			prompt: weatherAndEmailPrompt,
			initialization: {
				tools: [
					{
						...weatherTool,
						call: () => {
							throw new Error("the weather station is down");
						},
					},
					{ ...emailTool, call: () => "sent" },
				],
			},
		})
		.catch((error: unknown) => error);

	expect(failure).toMatchObject({
		code: "tool-call-failed",
		failures: [{ name: "get_weather", index: 0 }],
		generation: { success: true, type: "call" },
	});
});

test("skips tool invocation entirely for a failed generation", async () => {
	let invocations = 0;
	const runtime = await needle2({ provider: "direct" });
	runtimes.push(runtime);

	const result = await runtime.generate({
		prompt: weatherAndEmailPrompt,
		maxTokens: 4,
		initialization: {
			tools: [
				{
					...weatherTool,
					call: () => {
						invocations += 1;
						return "sunny";
					},
				},
				emailTool,
			],
		},
	});

	expect(result.success).toBe(false);
	expect(result.functionCalls).toEqual([]);
	expect(invocations).toBe(0);
});

test("leaves tool handlers uninvoked when generating a raw turn", async () => {
	let invocations = 0;
	const runtime = await needle2({ provider: "direct" });
	runtimes.push(runtime);

	const generation = await runtime.generateRaw({
		prompt: weatherAndEmailPrompt,
		initialization: {
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
		},
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
	const runtime = await needle2({ provider: "direct" });
	runtimes.push(runtime);

	const response = await runtime.runLoop({
		prompt: weatherAndEmailPrompt,
		initialization: {
			tools: [
				{ ...weatherTool, call: () => ({ celsius: 21, condition: "sunny" }) },
				{ ...emailTool, call: () => ({ status: "sent" }) },
			],
		},
	});

	expect(response.terminationCause).toBe("responded");
	expect(response.steps).toHaveLength(2);
	expect(response.steps[0]?.toolResponses).toEqual([
		{ celsius: 21, condition: "sunny" },
		{ status: "sent" },
	]);
	expect(response.steps[1]?.generation.functionCalls).toEqual([]);
	expect(response.steps[1]?.toolResponses).toEqual([]);
});

test("feeds a failed tool handler back to the model instead of throwing", async () => {
	const runtime = await needle2({ provider: "direct" });
	runtimes.push(runtime);

	const response = await runtime.runLoop({
		prompt: weatherAndEmailPrompt,
		initialization: {
			tools: [
				{
					...weatherTool,
					call: () => {
						throw new Error("the weather station is down");
					},
				},
				{ ...emailTool, call: () => ({ status: "sent" }) },
			],
		},
	});

	expect(response.steps[0]?.toolResponses).toEqual([
		{ error: "the weather station is down" },
		{ status: "sent" },
	]);
	expect(response.steps.length).toBeGreaterThan(1);
});

test("feeds an unknown tool error back for a tool declared without a handler", async () => {
	const runtime = await needle2({ provider: "direct" });
	runtimes.push(runtime);

	const response = await runtime.runLoop({
		prompt: weatherAndEmailPrompt,
		initialization: {
			tools: [{ ...weatherTool, call: () => ({ celsius: 21 }) }, emailTool],
		},
	});

	expect(response.steps[0]?.toolResponses).toEqual([
		{ celsius: 21 },
		{ error: "unknown tool: send_email" },
	]);
});

test("stops the loop once it runs out of turns", async () => {
	const runtime = await needle2({ provider: "direct" });
	runtimes.push(runtime);

	const response = await runtime.runLoop({
		prompt: weatherAndEmailPrompt,
		maximumTurns: 1,
		initialization: {
			tools: [
				{ ...weatherTool, call: () => ({ celsius: 21 }) },
				{ ...emailTool, call: () => ({ status: "sent" }) },
			],
		},
	});

	expect(response.terminationCause).toBe("maximum-turns-reached");
	expect(response.steps).toHaveLength(1);
});

test("rejects a turn count Needle 2 cannot support", async () => {
	const runtime = await needle2({ provider: "direct" });
	runtimes.push(runtime);

	await expect(
		runtime.runLoop({
			prompt: weatherAndEmailPrompt,
			maximumTurns: 9,
			initialization: {
				tools: [{ ...weatherTool, call: () => ({ celsius: 21 }) }],
			},
		}),
	).rejects.toMatchObject({ code: "invalid-turns" });
});

test("shares one direct model and requires reset between active runtimes", async () => {
	const thermostat = await needle2({ provider: "direct" });
	const weather = await needle2({ provider: "direct" });
	runtimes.push(thermostat, weather);

	const thermostatPrompt = {
		prompt: "set it to 21 degrees",
		initialization: thermostatInitialization,
	};
	const weatherPrompt = {
		prompt: "what's the weather in Paris?",
		initialization: weatherInitialization,
	};

	expect(
		(await thermostat.generate(thermostatPrompt)).functionCalls[0]?.name,
	).toBe("set_thermostat");
	await expect(weather.generate(weatherPrompt)).rejects.toMatchObject({
		code: "active-session",
	});
	await thermostat.reset();
	expect((await weather.generate(weatherPrompt)).functionCalls[0]?.name).toBe("get_weather");
	await weather.reset();
	expect(
		(await thermostat.generate(thermostatPrompt)).functionCalls[0]?.name,
	).toBe("set_thermostat");
});

describe.each([
	"direct",
	"worker",
] satisfies Needle2Provider[])("Needle2Runtime with the %s provider", (provider) => {
	test("generates a parsed response", async () => {
		const runtime = await needle2({ provider });
		runtimes.push(runtime);

		const result = await runtime.generate({
			prompt: "Send an email to blob@gmail.com asking them to go hiking.",
			initialization: emailInitialization,
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

	test("invokes tool handlers across the provider boundary", async () => {
		const runtime = await needle2({ provider });
		runtimes.push(runtime);

		const result = await runtime.generate({
			prompt: "set the thermostat to 21 degrees",
			initialization: {
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
			},
		});

		expect(result.success).toBe(true);
		if (!result.success) {
			throw new Error("Expected Needle 2 to call the thermostat tool.");
		}
		expect(result.functionCalls[0]?.output).toEqual({
			status: "ok",
			temperature: 21,
		});
	});

	test("rejects generation after disposal", async () => {
		const runtime = await needle2({ provider });
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

describe.each([
	"direct",
	"worker",
] satisfies Needle2Provider[])("Needle2Runtime auto engine with the %s provider", (provider) => {
	test("falls back to WASM when native artifacts are unavailable", async () => {
		const runtime = await needle2({ provider, engine: "auto" });
		runtimes.push(runtime);

		const result = await runtime.generate(thermostatRequest);

		expect(result.success).toBe(true);
		expect(result.functionCalls[0]?.name).toBe("set_thermostat");
	});
});
