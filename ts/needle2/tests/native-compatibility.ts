import assert from "node:assert/strict";
import test from "node:test";
import { needle2 } from "@edge-tools/needle2";
import {
	thermostatFunctionCalls,
	thermostatRequest,
	thermostatTools,
} from "./support.ts";

test("native addon generates a tool call", async () => {
	const first = await needle2({
		provider: "direct",
		engine: "native",
		tools: thermostatTools,
	});
	const second = await needle2({
		provider: "direct",
		engine: "native",
		tools: thermostatTools,
	});
	try {
		const firstResult = await first.generate(thermostatRequest);
		const secondResult = await second.generate(thermostatRequest);
		assert.equal(firstResult.success, true);
		assert.equal(secondResult.success, true);
		assert.deepEqual(firstResult.functionCalls, thermostatFunctionCalls);
		assert.deepEqual(secondResult.functionCalls, firstResult.functionCalls);
	} finally {
		await Promise.allSettled([first.dispose(), second.dispose()]);
	}
});

test("native addon works in a worker", async () => {
	const runtime = await needle2({
		provider: "worker",
		engine: "native",
		tools: thermostatTools,
	});
	try {
		const result = await runtime.generate(thermostatRequest);
		assert.equal(result.success, true);
		assert.deepEqual(result.functionCalls, thermostatFunctionCalls);
	} finally {
		await runtime.dispose();
	}
});
