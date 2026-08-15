import type {
	Needle2FunctionCall,
	Needle2GenerateOptions,
	Needle2Initialization,
} from "@edge-tools/needle2";

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

export const thermostatRequest: Needle2GenerateOptions = {
	prompt: "set the thermostat to 21 degrees",
	initialization: thermostatInitialization,
};

export const thermostatFunctionCalls: Needle2FunctionCall[] = [
	{
		name: "set_thermostat",
		arguments: { temperature: 21 },
	},
];
