import { needle2Runtime } from "./index";
import { setAssetBaseURL } from "./internal";

const script = typeof document === "undefined" ? undefined : document.currentScript;
if (script && "src" in script && script.src) {
  setAssetBaseURL(new URL("./", script.src));
} else if (typeof location !== "undefined") {
  setAssetBaseURL(new URL("./", location.href));
}

export default needle2Runtime;
