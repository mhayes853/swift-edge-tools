import assert from "node:assert/strict";
import test from "node:test";
import { needle2Runtime } from "@edge-tools/needle2";
import { thermostatFunctionCalls, thermostatRequest } from "./support.ts";

const providers = ["direct", "worker"] as const;

for (const provider of providers) {
	test(`${provider} provider generates a tool call`, async () => {
		const runtime = await needle2Runtime({ provider });
		try {
			const result = await runtime.generate(thermostatRequest);

			assert.equal(result.success, true);
			assert.equal(result.type, "call");
			assert.deepEqual(result.functionCalls, thermostatFunctionCalls);
		} finally {
			await runtime.dispose();
		}
	});
}
