// Gemma 4 uses the Hub's `tokenizer.json` BPE tokenizer rather than Qwen's
// GPT-2 byte-BPE files.  Keeping this implementation separate prevents a
// model-selection flag from silently producing Qwen token IDs for Gemma.
import { GEMMA_E4B_CFG } from './gemma-config.js';

// Keep tokenizer assets pinned to the exact checkpoint revision used by the
// weight loader.  Mixing a future tokenizer revision with a cached weight
// schema can silently change special-token IDs and corrupt multimodal prompts.
const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

async function fetchCached(config, name, { optional = false } = {}) {
  const url = `https://huggingface.co/${config.repo}/resolve/${config.revision}/` + name;
  const cache = await caches.open(`${config.cacheNamespace}-tokenizer-v1`).catch(() => null);
  const hit = cache && await cache.match(url);
  if (hit) return hit.text();
  const res = await fetch(url);
  if (!res.ok) {
    if (optional) return null;
    throw new Error(`fetch ${name}: ${res.status}`);
  }
  if (cache) await cache.put(url, res.clone()).catch(() => {});
  return res.text();
}

/**
 * Browser-compatible implementation of the simple BPE graph serialized in
 * Gemma 4's tokenizer.json.  The tokenizer's normalizer turns literal spaces
 * into SentencePiece-style U+2581 markers; no native/WASM tokenizer is needed.
 */
export class GemmaTokenizer {
  static async load(status = () => {}, modelConfig = GEMMA_E4B_CFG) {
    status('Loading Gemma tokenizer…');
    const [tokenizerText, configText, templateText] = await Promise.all([
      fetchCached(modelConfig, 'tokenizer.json'),
      fetchCached(modelConfig, 'tokenizer_config.json'),
      fetchCached(modelConfig, 'chat_template.jinja', { optional: true }),
    ]);
    const tokenizer = JSON.parse(tokenizerText);
    const config = JSON.parse(configText);
    if (tokenizer.model?.type !== 'BPE' || !tokenizer.model?.vocab || !tokenizer.model?.merges) {
      throw new Error('Unexpected Gemma tokenizer format; expected tokenizer.json BPE metadata.');
    }

    const t = new GemmaTokenizer();
    t.vocab = new Map(Object.entries(tokenizer.model.vocab).map(([token, id]) => [token, Number(id)]));
    t.idToToken = new Map([...t.vocab].map(([token, id]) => [id, token]));
    t.ranks = new Map();
    for (let i = 0; i < tokenizer.model.merges.length; i++) {
      const pair = tokenizer.model.merges[i];
      // Tokenizers JSON encodes BPE merges as [left, right] pairs.
      if (Array.isArray(pair) && pair.length === 2) t.ranks.set(`${pair[0]}\u0000${pair[1]}`, i);
    }

    t.special = new Map();
    t.specialById = new Map();
    for (const entry of tokenizer.added_tokens || []) {
      if (typeof entry?.content === 'string' && Number.isInteger(entry?.id)) {
        t.special.set(entry.content, entry.id);
        t.specialById.set(entry.id, entry.content);
      }
    }
    // A few control tokens appear in vocab on older Hub revisions rather than
    // `added_tokens`; accepting both keeps cached tokenizer revisions usable.
    for (const token of [
      '<pad>', '<eos>', '<bos>', '<unk>', '<|turn>', '<turn|>', '<|channel>',
      '<channel|>', '<|think|>', '<|image>', '<image|>', '<|image|>',
      '<|audio>', '<audio|>', '<|audio|>', '<|video|>',
      '<|tool>', '<tool|>', '<|tool_call>', '<tool_call|>',
      '<|tool_response>', '<tool_response|>', '<|"|>',
    ]) {
      const id = t.vocab.get(token);
      if (id !== undefined) {
        t.special.set(token, id);
        t.specialById.set(id, token);
      }
    }
    t.cache = new Map();
    t.eos = t.special.get(config.eos_token || '<eos>') ?? 1;
    t.bos = t.special.get(config.bos_token || '<bos>') ?? 2;
    t.pad = t.special.get(config.pad_token || '<pad>') ?? 0;
    t.image = t.special.get(config.image_token || '<|image|>') ?? 258880;
    t.audio = t.special.get(config.audio_token || '<|audio|>') ?? 258881;
    t.video = t.special.get('<|video|>') ?? 258884;
    t.turnEnd = t.special.get(config.eot_token || '<turn|>') ?? 106;
    t.toolCallOpen = t.special.get('<|tool_call>');
    t.toolCallClose = t.special.get('<tool_call|>');
    t.chatTemplate = templateText ?? config.chat_template ?? null;
    return t;
  }

  #specialPattern() {
    if (!this._specialPattern) {
      const values = [...this.special.keys()].sort((a, b) => b.length - a.length);
      this._specialPattern = values.map(escapeRegExp).join('|');
    }
    return this._specialPattern;
  }

  /** Escape reserved Gemma control tokens in untrusted text and tool values. */
  sanitize(text) {
    const source = this.#specialPattern();
    if (!source) return text;
    return text.replace(new RegExp(source, 'g'), (match) => match[0] + '\u200b' + match.slice(1));
  }

  #fallback(symbol) {
    const out = [];
    const bytes = new TextEncoder().encode(symbol);
    for (const byte of bytes) {
      const token = `<0x${byte.toString(16).toUpperCase().padStart(2, '0')}>`;
      const id = this.vocab.get(token);
      if (id === undefined) return [this.special.get('<unk>') ?? 3];
      out.push(id);
    }
    return out;
  }

  #bpe(piece) {
    const cached = this.cache.get(piece);
    if (cached) return cached;
    let symbols = Array.from(piece);
    while (symbols.length > 1) {
      let best = -1;
      let bestRank = Number.POSITIVE_INFINITY;
      for (let i = 0; i + 1 < symbols.length; i++) {
        const rank = this.ranks.get(`${symbols[i]}\u0000${symbols[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          best = i;
        }
      }
      if (best < 0) break;
      symbols = [...symbols.slice(0, best), symbols[best] + symbols[best + 1], ...symbols.slice(best + 2)];
    }
    const ids = symbols.flatMap((symbol) => {
      const id = this.vocab.get(symbol);
      return id === undefined ? this.#fallback(symbol) : [id];
    });
    if (this.cache.size < 50_000) this.cache.set(piece, ids);
    return ids;
  }

  /** Encode raw prompt text. The caller inserts `<bos>` exactly once. */
  encode(text) {
    const source = this.#specialPattern();
    const parts = source ? text.split(new RegExp(`(${source})`, 'g')) : [text];
    const ids = [];
    for (const part of parts) {
      if (!part) continue;
      const special = this.special.get(part);
      if (special !== undefined) {
        ids.push(special);
        continue;
      }
      // Gemma's tokenizer.json normalizer maps literal spaces to the
      // SentencePiece marker before BPE merges are applied.
      ids.push(...this.#bpe(part.replaceAll(' ', '▁')));
    }
    return ids;
  }

  makeDecoder() {
    const decoder = new TextDecoder('utf-8', { fatal: false });
    const flush = (bytes) => bytes.length ? decoder.decode(new Uint8Array(bytes), { stream: true }) : '';
    let bytes = [];
    return (id) => {
      const special = this.specialById.get(id);
      if (special !== undefined) return flush(bytes.splice(0)) + special;
      const token = this.idToToken.get(id);
      if (token === undefined) return '';
      const m = /^<0x([0-9A-F]{2})>$/.exec(token);
      if (m) {
        bytes.push(Number.parseInt(m[1], 16));
        return '';
      }
      return flush(bytes.splice(0)) + token.replaceAll('▁', ' ');
    };
  }

  vocabSize() { return this.idToToken.size; }

  specialTokenId(text) { return this.special.get(text); }

  /** Plain text of a regular token, for constraint character classification.
   * Control/special tokens and byte-fallback tokens return undefined. */
  tokenText(id) {
    if (this.specialById.has(id)) return undefined;
    const token = this.idToToken.get(id);
    if (token === undefined || /^<0x[0-9A-F]{2}>$/.test(token)) return undefined;
    return token.replaceAll('▁', ' ');
  }
}
