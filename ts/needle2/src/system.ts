export type Needle2SystemFactValue = string | number | boolean;

export type Needle2SystemValues = {
	[key: string]: Needle2SystemFactValue | undefined;
	date?: string;
	locale?: string;
	device?: string;
	battery?: string;
	network?: string;
	location?: string;
	user?: string;
	assistant?: string;
};

export type Needle2SystemFactProvider = () =>
	| Needle2SystemFactValue
	| null
	| undefined
	| PromiseLike<Needle2SystemFactValue | null | undefined>;

export type Needle2SystemFactOverrides = Readonly<
	Record<string, Needle2SystemFactValue | null>
>;

export type Needle2SystemFactProviders = Readonly<
	Record<string, Needle2SystemFactProvider>
>;

export type Needle2SystemFactsProvider = () =>
	| Needle2SystemValues
	| PromiseLike<Needle2SystemValues>;

export type Needle2SystemValuesOptions = {
	overrides?: Needle2SystemFactOverrides;
	providers?: Needle2SystemFactProviders;
	includeLocation?: boolean;
};

const DEFAULT_SYSTEM_KEYS = [
	"date",
	"locale",
	"device",
	"battery",
	"network",
	"location",
	"user",
	"assistant",
] as const;

export function defaultSystemPrompt(facts: Needle2SystemValues = {}): string {
	const values = new Map<string, Needle2SystemFactValue>();
	for (const key of DEFAULT_SYSTEM_KEYS) {
		const value = facts[key];
		if (value != null) {
			values.set(key, value);
		}
	}
	for (const [key, value] of Object.entries(facts)) {
		if (
			!DEFAULT_SYSTEM_KEYS.includes(
				key as (typeof DEFAULT_SYSTEM_KEYS)[number],
			) &&
			value != null
		) {
			values.set(key, value);
		}
	}
	return [...values]
		.map(([key, value]) => `${key}: ${String(value)}`)
		.join("; ");
}

export async function defaultSystemValues(
	options: Needle2SystemValuesOptions = {},
): Promise<Needle2SystemValues> {
	const providers: Record<string, Needle2SystemFactProvider> = {
		...DEFAULT_ENVIRONMENT_PROVIDERS,
		...(options.includeLocation ? { location: environmentLocation } : {}),
		...options.providers,
	};
	const overrides = options.overrides ?? {};
	const keys = new Set([...Object.keys(providers), ...Object.keys(overrides)]);
	const facts: Needle2SystemValues = {};

	const values = await Promise.all(
		[...keys].map(async (key) => {
			const override = overrides[key];
			const value =
				override !== undefined
					? override
					: providers[key]
						? await providers[key]()
						: undefined;
			return [key, value] as const;
		}),
	);
	for (const [key, value] of values) {
		if (value != null) {
			facts[key] = value;
		}
	}
	return facts;
}

const DEFAULT_ENVIRONMENT_PROVIDERS: Record<string, Needle2SystemFactProvider> =
	{
		date: currentDate,
		locale: currentLocale,
		device: currentDevice,
		battery: currentBattery,
		network: currentNetwork,
	};

function currentDate(): string {
	const parts = new Intl.DateTimeFormat("en-US", {
		weekday: "short",
		year: "numeric",
		month: "2-digit",
		day: "2-digit",
		hour: "2-digit",
		minute: "2-digit",
		hourCycle: "h23",
	}).formatToParts(new Date());
	const values = Object.fromEntries(
		parts.map((part) => [part.type, part.value]),
	);
	return `${values.year}-${values.month}-${values.day} ${values.weekday} ${values.hour}:${values.minute}`;
}

function currentLocale(): string {
	if (typeof navigator !== "undefined" && navigator.language) {
		return navigator.language;
	}
	return new Intl.DateTimeFormat().resolvedOptions().locale;
}

function currentDevice(): string | undefined {
	if (typeof navigator === "undefined") {
		return undefined;
	}
	const userAgent = navigator.userAgent.toLowerCase();
	if (/ipad|tablet|playbook|silk/.test(userAgent)) {
		return "tablet";
	}
	if (/mobi|android|iphone|ipod|phone/.test(userAgent)) {
		return "phone";
	}
	return "desktop";
}

async function currentBattery(): Promise<string | undefined> {
	if (typeof navigator === "undefined") {
		return undefined;
	}
	const batteryNavigator = navigator as Navigator & {
		getBattery?: () => Promise<{ level: number }>;
	};
	if (!batteryNavigator.getBattery) {
		return undefined;
	}
	try {
		const battery = await batteryNavigator.getBattery();
		return `${Math.round(battery.level * 100)}%`;
	} catch {
		return undefined;
	}
}

function currentNetwork(): string | undefined {
	if (typeof navigator === "undefined") {
		return undefined;
	}
	const networkNavigator = navigator as Navigator & {
		connection?: {
			type?: string;
			effectiveType?: string;
		};
	};
	return (
		networkNavigator.connection?.type ??
		networkNavigator.connection?.effectiveType ??
		(navigator.onLine === false ? "offline" : "online")
	);
}

function environmentLocation(): Promise<string | undefined> {
	if (typeof navigator === "undefined" || !navigator.geolocation) {
		return Promise.resolve(undefined);
	}
	return new Promise((resolve) => {
		navigator.geolocation.getCurrentPosition(
			(position) => {
				resolve(`${position.coords.latitude}, ${position.coords.longitude}`);
			},
			() => resolve(undefined),
			{ maximumAge: 300_000, timeout: 5_000 },
		);
	});
}
