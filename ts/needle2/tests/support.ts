import type {
	Needle2FunctionCall,
	Needle2GenerateOptions,
} from "@edge-tools/needle2";

export const thermostatTools = [
	{
		name: "set_thermostat",
		description: "Set the thermostat temperature.",
		parameters: {
			type: "object",
			properties: { temperature: { type: "integer" } },
			required: ["temperature"],
		},
	},
] as const;

export const thermostatRequest: Needle2GenerateOptions = {
	prompt: "set the thermostat to 21 degrees",
};

export const thermostatFunctionCalls: Needle2FunctionCall[] = [
	{
		name: "set_thermostat",
		arguments: { temperature: 21 },
	},
];
