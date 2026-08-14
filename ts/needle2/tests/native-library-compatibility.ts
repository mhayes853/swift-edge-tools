import assert from "node:assert/strict";
import test from "node:test";
import { needle2Runtime } from "../dist/index.js";

for (const provider of ["direct", "worker"] as const) {
	test(`${provider} C ABI native library generates a tool call`, async () => {
		const runtime = await needle2Runtime({
			provider,
			engine: "native",
		});
		try {
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
						},
					],
				},
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
