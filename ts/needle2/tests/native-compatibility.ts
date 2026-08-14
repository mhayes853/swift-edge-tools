import assert from "node:assert/strict";
import test from "node:test";
import { needle2Runtime } from "../dist/index.js";

const initialization = {
	tools: [
		{
			name: "set_thermostat",
			description: "Set the thermostat temperature.",
			parameters: {
				type: "object",
				properties: { temperature: { type: "integer" } },
				required: ["temperature"],
			},
		},
	],
};

test("native addon generates a tool call", async () => {
	const first = await needle2Runtime({
		provider: "direct",
		engine: "native",
	});
	const second = await needle2Runtime({
		provider: "direct",
		engine: "native",
	});
	try {
		const prompt = { prompt: "set the thermostat to 21 degrees", initialization };
		const firstResult = await first.generate(prompt);
		const secondResult = await second.generate(prompt);
		assert.equal(firstResult.success, true);
		assert.equal(secondResult.success, true);
		assert.deepEqual(firstResult.functionCalls, [
			{
				name: "set_thermostat",
				arguments: { temperature: 21 },
			},
		]);
		assert.deepEqual(secondResult.functionCalls, firstResult.functionCalls);
	} finally {
		await Promise.all([first.dispose(), second.dispose()]);
	}
});

test("native addon works in a worker", async () => {
	const runtime = await needle2Runtime({
		provider: "worker",
		engine: "native",
	});
	try {
		const result = await runtime.generate({
			prompt: "set the thermostat to 21 degrees",
			initialization,
		});
		assert.equal(result.success, true);
		assert.deepEqual(result.functionCalls, [
			{
				name: "set_thermostat",
				arguments: { temperature: 21 },
			},
		]);
	} finally {
		await runtime.dispose();
	}
});
