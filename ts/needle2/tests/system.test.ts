import { describe, expect, test } from "vitest";
import { defaultSystemValues, defaultSystemPrompt } from "@edge-tools/needle2";

describe("Needle2 system facts tests", () => {
	test("formats standard and arbitrary facts in a stable order", () => {
		expect(
			defaultSystemPrompt({
				assistant: "Needle",
				date: "2026-07-21 Tue 14:30",
				"account tier": "pro",
				locale: "en-US",
			}),
		).toBe(
			"date: 2026-07-21 Tue 14:30; locale: en-US; assistant: Needle; account tier: pro",
		);
	});

	test("lets overrides suppress and replace environment facts", async () => {
    const facts = await defaultSystemValues({
      overrides: {
				date: "tomorrow",
				device: null,
				"account tier": "pro",
			},
			providers: {
				date: () => "today",
        device: () => "phone",
				"account tier": async () => "free",
			},
		});

		expect(facts.date).toBe("tomorrow");
		expect(facts.device).toBeUndefined();
		expect(facts["account tier"]).toBe("pro");
	});

	test("allows location to be supplied explicitly", async () => {
		const facts = await defaultSystemValues({
			providers: {
				location: () => "injected location",
			},
		});

		expect(facts.location).toBe("injected location");
	});
});
