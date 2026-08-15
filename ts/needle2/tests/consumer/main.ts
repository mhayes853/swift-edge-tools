import { needle2 } from "@edge-tools/needle2";

globalThis.createNeedle2Runtime = needle2;

declare global {
  var createNeedle2Runtime: typeof needle2;
}
