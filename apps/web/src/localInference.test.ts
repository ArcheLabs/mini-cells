import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { MiniCellsLocalRuntime } from "./localInference";

const wasm = new Uint8Array(readFileSync(resolve(process.cwd(), "../../target/wasm32-unknown-unknown/release/minicells_wasm.wasm")));
const model = new Uint8Array(readFileSync(resolve(process.cwd(), "../../service/generated/genesis_model.bin")));
const modelHash = "0x0af6953731041b3612fdcb3cc481a09d501d9bfe4d8401279d02831232ff2fd2";

describe("browser WASM local inference", () => {
  it("loads the real artifact and is deterministic", async () => {
    const runtime = new MiniCellsLocalRuntime(); await runtime.init(wasm); runtime.setModel({ generation: 0, modelHash, modelBytes: model });
    expect(runtime.infer("hello")).toEqual(runtime.infer("hello"));
  });
  it("rejects wrong hash, length, and vocabulary", async () => {
    const runtime = new MiniCellsLocalRuntime(); await runtime.init(wasm);
    expect(() => runtime.setModel({ generation: 0, modelHash: "0x" + "00".repeat(32), modelBytes: model })).toThrow(/integrity/);
    expect(() => runtime.setModel({ generation: 0, modelHash, modelBytes: model.slice(0, -1) })).toThrow(/length/);
    runtime.setModel({ generation: 0, modelHash, modelBytes: model }); expect(() => runtime.infer("@bad")).toThrow(/unsupported/);
  });
});
