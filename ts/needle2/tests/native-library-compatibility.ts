import assert from "node:assert/strict";
import test from "node:test";
import { needle2Runtime } from "@edge-tools/needle2";
import { thermostatInitialization } from "./support.ts";

for (const provider of ["direct", "worker"] as const) {
	test(`${provider} C ABI native library generates a tool call`, async () => {
		const runtime = await needle2Runtime({
			provider,
			engine: "native",
		});
		try {
			const result = await runtime.generate({
				prompt: "set the thermostat to 21 degrees",
				initialization: thermostatInitialization,
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
}
