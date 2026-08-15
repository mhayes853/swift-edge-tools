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

test("isolates multiple direct runtimes sharing one native module", async () => {
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
	expect((await weather.generate(weatherPrompt)).functionCalls[0]?.name).toBe(
		"get_weather",
	);
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
