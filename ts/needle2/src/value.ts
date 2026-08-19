export type Needle2JSONPrimitive = string | number | boolean | null;

export type Needle2JSONValue =
	| Needle2JSONPrimitive
	| readonly Needle2JSONValue[]
	| Needle2JSONObject;

export type Needle2JSONObject = {
	[key: string]: Needle2JSONValue;
};
