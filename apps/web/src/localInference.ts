export type LocalModel = { generation: number; modelHash: string; modelBytes: Uint8Array };

const MODEL_BYTES = 8952;
const SYMBOLS = "abcdefghijklmnopqrstuvwxyz0123456789 .,?!'-";
const STATUS: Record<number, string> = { 1: "invalid model length", 2: "model hash integrity check failed", 3: "input is longer than 32 bytes", 4: "input contains an unsupported character", 5: "model decoding failed", 6: "WASM inference failed" };

export class MiniCellsLocalRuntime {
  private instance?: WebAssembly.Instance;
  private model?: LocalModel;
  async init(source: string | Uint8Array = "/minicells_core.wasm"): Promise<void> {
    if (this.instance) return;
    const bytes = typeof source === "string" ? new Uint8Array(await (await fetch(source)).arrayBuffer()) : source;
    const result = await WebAssembly.instantiate(bytes, {});
    const candidate = result as unknown as { instance?: WebAssembly.Instance };
    this.instance = candidate.instance || result as WebAssembly.Instance;
  }
  setModel(model: LocalModel): void {
    if (!this.instance) throw new Error("local WASM runtime is not initialized");
    if (model.modelBytes.byteLength !== MODEL_BYTES) throw new Error("invalid model length");
    this.model = { ...model, modelBytes: new Uint8Array(model.modelBytes) };
    try { this.verifyModel(); } catch (error) { this.model = undefined; throw error; }
  }
  infer(text: string): { output: string; generation: number; modelHash: string } {
    if (!this.instance || !this.model) throw new Error("a verified model is not loaded");
    const bytes = new TextEncoder().encode(text);
    if (bytes.length > 32) throw new Error("input is longer than 32 bytes");
    const exports = this.exports();
    new Uint8Array(this.memory().buffer, exports.minicells_input_ptr(), 32).fill(0);
    new Uint8Array(this.memory().buffer, exports.minicells_input_ptr(), bytes.length).set(bytes);
    const status = exports.minicells_infer(MODEL_BYTES, bytes.length);
    if (status !== 0) throw new Error(STATUS[status] || `WASM status ${status}`);
    const length = exports.minicells_output_len();
    const ids = new Uint8Array(this.memory().buffer, exports.minicells_output_ptr(), length);
    let output = "";
    for (const id of ids) { if (id > 0) output += SYMBOLS[id - 1] || ""; }
    return { output, generation: this.model.generation, modelHash: this.model.modelHash };
  }
  private verifyModel(): void { const model = this.model!; const exports = this.exports(); new Uint8Array(this.memory().buffer, exports.minicells_model_ptr(), MODEL_BYTES).set(model.modelBytes); const hash = hexToBytes(model.modelHash); if (hash.length !== 32) throw new Error("invalid model hash"); new Uint8Array(this.memory().buffer, exports.minicells_hash_ptr(), 32).set(hash); const status = exports.minicells_infer(MODEL_BYTES, 0); if (status !== 0) throw new Error(STATUS[status] || `WASM status ${status}`); }
  private memory(): WebAssembly.Memory { const memory = this.exports().memory; if (!(memory instanceof WebAssembly.Memory)) throw new Error("WASM memory export missing"); return memory; }
  private exports(): Record<string, any> { if (!this.instance) throw new Error("local WASM runtime is not initialized"); return this.instance.exports as Record<string, any>; }
}
function hexToBytes(value: string): Uint8Array { const raw = value.replace(/^0x/, ""); if (raw.length % 2) throw new Error("invalid hex"); const out = new Uint8Array(raw.length / 2); for (let i = 0; i < out.length; i++) out[i] = Number.parseInt(raw.slice(i * 2, i * 2 + 2), 16); return out; }
