import type { ConstraintTokenizer } from "../../lib/webgpu-llm/tool-constraint";

type TokenSpec = { text: string; special: boolean };

export class FakeTokenizer implements ConstraintTokenizer {
  private readonly specs: TokenSpec[] = [];
  private readonly ids = new Map<string, number>();
  private readonly specials = new Map<string, number>();

  constructor() {
    for (const text of ['<|"|>', "<tool_call|>", "<eos>", "<turn>", "<byte>"]) {
      this.add(text, true);
    }
    for (const text of [
      "abcd", "abc", "ab", "23", "20", "19", "11", "10", ".5",
      "false", "true", "null",
    ]) {
      this.add(text, false);
    }
    for (let code = 32; code <= 126; code++) this.add(String.fromCharCode(code), false);
  }

  private add(text: string, special: boolean): void {
    if (this.ids.has(text)) return;
    const id = this.specs.length;
    this.specs.push({ text, special });
    this.ids.set(text, id);
    if (special) this.specials.set(text, id);
  }

  id(text: string): number {
    const id = this.ids.get(text);
    if (id === undefined) throw new Error(`Unknown fake token: ${text}`);
    return id;
  }

  encode(text: string): number[] {
    const ids: number[] = [];
    let offset = 0;
    const candidates = this.specs
      .map((spec, id) => ({ ...spec, id }))
      .sort((a, b) => b.text.length - a.text.length || a.id - b.id);
    while (offset < text.length) {
      const match = candidates.find(({ text: token }) => text.startsWith(token, offset));
      if (!match) throw new Error(`Fake tokenizer cannot encode ${JSON.stringify(text.slice(offset))}`);
      ids.push(match.id);
      offset += match.text.length;
    }
    return ids;
  }

  vocabSize(): number {
    return this.specs.length;
  }

  specialTokenId(text: string): number | undefined {
    return this.specials.get(text);
  }

  tokenText(id: number): string | undefined {
    const spec = this.specs[id];
    return spec && !spec.special ? spec.text : undefined;
  }
}
