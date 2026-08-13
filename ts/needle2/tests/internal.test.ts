import { describe, expect, test } from "vitest";
import { PromiseQueue } from "../src/internal";

describe("PromiseQueue tests", () => {
  test("serializes operations and continues after an error", async () => {
    const queue = new PromiseQueue();
    const events: string[] = [];
    let releaseFirst!: () => void;
    const firstGate = new Promise<void>(resolve => {
      releaseFirst = resolve;
    });
    const first = queue.enqueue(async () => {
      events.push("first started");
      await firstGate;
      events.push("first finished");
    });
    const second = queue.enqueue(() => {
      events.push("second");
      throw new Error("expected failure");
    });
    const third = queue.enqueue(() => {
      events.push("third");
    });

    await Promise.resolve();
    expect(events).toEqual(["first started"]);
    releaseFirst();
    await first;
    await expect(second).rejects.toThrow("expected failure");
    await third;
    expect(events).toEqual(["first started", "first finished", "second", "third"]);
  });
});
