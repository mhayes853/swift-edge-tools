# Needle 2 TS

Typescript package for Needle 2. Supports browser and node/deno/bun environments.

## Usage

### Basic

```ts
import { needle2 } from "@edge-tools/needle2"

const getWeather = {
	name: "get_weather",
	description: "Get the weather for a city.",
	parameters: {
		type: "object",
		properties: {
			city: { type: "string" }
		},
		required: ["city"]
	},
  call: async ({ city }: { city: string }) => {
    // Fetch the weather from somewhere...
  }
}

const needle = await needle2({ 
  provider: "direct" // Can also be "worker" if you want inference to run in a web worker.
})

const results = await needle.generate({
  prompt: "What is the weather in San Francisco?",
  initialization: { tools: [getWeather] }
})
console.log(results.functionCalls[0].output)
```

### Zod

```ts
import { zodTool } from "@edge-tools/needle2/zod"
import { z } from "zod"

const getWeather = zodTool({
	name: "get_weather",
	description: "Get the weather for a city.",
	parameters: z.object({ city: z.string() }),
  call: async ({ city }) => {
    // Fetch the weather from somewhere...
  }
})
```

### With System Values

```ts
import { defaultSystemValues, defaultSystemPrompt } from "@edge-tools/needle2"

// Loads all the default keys (eg. Battery life, location, etc.)
const values = await defaultSystemValues()

// Also can provide overrides and custom factories for various environments.
const values = await defaultSystemValues({
  includeLocation: true, // false by default, will request location permissions if in browser
  overrides: {
    date: "tomorrow",
  },
  providers: {
    battery: myEnvBatteryPercentageLoader
  }
})

const results = await needle.generate({
  prompt: "What is the weather in San Francisco?",
  initialization: { 
    tools: [getWeather],
    systemPrompt: defaultSystemPrompt(values)
  }
})
```
