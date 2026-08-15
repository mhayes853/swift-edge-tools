import type { Needle2Initialization } from "@edge-tools/needle2";

export const thermostatInitialization: Needle2Initialization = {
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
