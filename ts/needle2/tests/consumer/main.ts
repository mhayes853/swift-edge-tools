import { needle2Runtime } from "@edge-tools/needle2";

globalThis.createNeedle2Runtime = needle2Runtime;

declare global {
  var createNeedle2Runtime: typeof needle2Runtime;
}
