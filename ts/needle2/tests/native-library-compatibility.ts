import assert from "node:assert/strict";
import test from "node:test";
import { needle2Runtime } from "@edge-tools/needle2";
import { thermostatFunctionCalls, thermostatRequest } from "./support.ts";

for (const provider of ["direct", "worker"] as const) {
	test(`${provider} C ABI native library generates a tool call`, async () => {
		const runtime = await needle2Runtime({
			provider,
			engine: "native",
		});
		try {
			const result = await runtime.generate(thermostatRequest);
			assert.equal(result.success, true);
			assert.deepEqual(result.functionCalls, thermostatFunctionCalls);
		} finally {
			await runtime.dispose();
		}
	});
}
