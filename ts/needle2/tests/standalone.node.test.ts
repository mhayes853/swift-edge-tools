import { createServer, type Server } from "node:http";
import { readFile } from "node:fs/promises";
import type { AddressInfo } from "node:net";
import { resolve } from "node:path";
import { chromium, type Browser } from "playwright";
import { afterEach, expect, test } from "vitest";

const browsers: Browser[] = [];
const servers: Server[] = [];

afterEach(async () => {
  await Promise.allSettled(browsers.splice(0).map((browser) => browser.close()));
  await Promise.allSettled(servers.splice(0).map(closeServer));
});

test("an ordinary HTML page runs with one standalone JavaScript file", async () => {
  const server = createStandaloneServer();
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));

  const browser = await chromium.launch({ headless: true });
  browsers.push(browser);
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${serverAddress(server).port}`);

  const result = await page.evaluate(() => globalThis.needle2Result);
  expect(result).toEqual({ success: true, functionName: "set_thermostat" });
});

function createStandaloneServer(): Server {
  const files = new Map<string, readonly [string, string]>([
    ["/", ["tests/standalone/index.html", "text/html; charset=utf-8"]],
    ["/needle2.min.js", ["dist/needle2.min.js", "text/javascript; charset=utf-8"]],
    ["/needle.wasm", ["dist/needle.wasm", "application/wasm"]],
    ["/needle2.cact", ["dist/needle2.cact", "application/octet-stream"]]
  ]);
  return createServer(async (request, response) => {
    const file = files.get(request.url ?? "");
    if (!file) {
      response.writeHead(404).end();
      return;
    }
    try {
      const contents = await readFile(resolve(import.meta.dirname, "..", file[0]));
      response.writeHead(200, { "content-type": file[1] }).end(contents);
    } catch (error) {
      response.writeHead(500).end(String(error));
    }
  });
}

function serverAddress(server: Server): AddressInfo {
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("The standalone test server did not bind to a TCP address.");
  }
  return address;
}

function closeServer(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.close((error) => error ? reject(error) : resolve());
  });
}

declare global {
  var needle2Result: Promise<
    | { success: boolean; functionName?: string }
    | { error: string }
  >;
}
