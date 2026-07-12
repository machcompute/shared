// Byte-level BPE tokenizer (Qwen2Tokenizer-compatible), built from
// vocab.json + merges.txt + tokenizer_config.json fetched from the repo.
import { CFG } from './config.js';

const HF = `https://huggingface.co/${CFG.repo}/resolve/main/`;

// GPT-2 bytes<->unicode bijection
function byteMaps() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const b2u = new Array(256), u2b = new Map();
  for (let i = 0; i < bs.length; i++) { b2u[bs[i]] = String.fromCharCode(cs[i]); u2b.set(cs[i], bs[i]); }
  return { b2u, u2b };
}

// Qwen pretokenize regex, converted to JS (inline (?i:) group expanded).
const PRETOK = /'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

async function fetchCached(name) {
  // small files → Cache API keyed by URL
  const url = HF + name;
  const cache = await caches.open('qwen35-tokenizer').catch(() => null);
  if (cache) {
    const hit = await cache.match(url);
    if (hit) return hit.text();
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${name}: ${res.status}`);
  if (cache) await cache.put(url, res.clone()).catch(() => {});
  return res.text();
}

export class Tokenizer {
  static async load(status = () => {}) {
    status('Loading tokenizer…');
    const [vocabText, mergesText, cfgText] = await Promise.all([
      fetchCached('vocab.json'), fetchCached('merges.txt'), fetchCached('tokenizer_config.json'),
    ]);
    const t = new Tokenizer();
    t.vocab = new Map(Object.entries(JSON.parse(vocabText)));
    t.ranks = new Map();
    const lines = mergesText.split('\n');
    let rank = 0;
    for (const line of lines) {
      if (!line || line.startsWith('#version')) continue;
      t.ranks.set(line, rank++); // "a b" as-is
    }
    const tcfg = JSON.parse(cfgText);
    t.special = new Map();  // string -> id
    t.specialById = new Map();
    for (const [id, tok] of Object.entries(tcfg.added_tokens_decoder || {})) {
      t.special.set(tok.content, Number(id));
      t.specialById.set(Number(id), tok.content);
    }
    t.idToTok = new Array(CFG.vocab);
    for (const [tok, id] of t.vocab) t.idToTok[id] = tok;
    const { b2u, u2b } = byteMaps();
    t.b2u = b2u; t.u2b = u2b;
    t.cache = new Map();
    t.eos = t.special.get(CFG.eosText);
    t.utf8enc = new TextEncoder();
    t.chatTemplate = tcfg.chat_template;
    return t;
  }

  #bpe(piece) {
    const hit = this.cache.get(piece);
    if (hit) return hit;
    let word = Array.from(piece); // unicode-mapped byte chars
    while (word.length > 1) {
      let best = null, bestRank = Infinity, bestI = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const r = this.ranks.get(word[i] + ' ' + word[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; bestI = i; best = word[i] + word[i + 1]; }
      }
      if (bestI < 0) break;
      word = [...word.slice(0, bestI), best, ...word.slice(bestI + 2)];
    }
    const ids = word.map((w) => {
      const id = this.vocab.get(w);
      if (id === undefined) throw new Error('token not in vocab: ' + JSON.stringify(w));
      return id;
    });
    if (this.cache.size < 50000) this.cache.set(piece, ids);
    return ids;
  }

  #specialsPattern() {
    if (!this._specialsSrc) {
      const specials = [...this.special.keys()].sort((a, b) => b.length - a.length);
      this._specialsSrc = specials.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
    }
    return this._specialsSrc;
  }

  /** Break special-token strings inside untrusted text (user input, tool
   * results) so encode() tokenizes them as plain bytes instead of forging
   * control tokens like <|im_end|> or </tool_response>. */
  sanitize(text) {
    const re = new RegExp(this.#specialsPattern(), 'g');
    return text.replace(re, (m) => m[0] + '\u200b' + m.slice(1));
  }

  encode(text) {
    // split on special tokens first
    const splitRe = new RegExp('(' + this.#specialsPattern() + ')', 'g');
    const out = [];
    for (const part of text.split(splitRe)) {
      if (!part) continue;
      const sid = this.special.get(part);
      if (sid !== undefined) { out.push(sid); continue; }
      for (const m of part.match(PRETOK) || []) {
        const bytes = this.utf8enc.encode(m);
        let mapped = '';
        for (const b of bytes) mapped += this.b2u[b];
        out.push(...this.#bpe(mapped));
      }
    }
    return out;
  }

  /** Streaming decoder: returns text as it becomes valid UTF-8. */
  makeDecoder() {
    const dec = new TextDecoder('utf-8', { fatal: false });
    return (id) => {
      const sp = this.specialById.get(id);
      if (sp !== undefined) { return dec.decode(new Uint8Array(0)) + sp; }
      const tok = this.idToTok[id];
      if (tok === undefined) return '';
      const bytes = new Uint8Array(tok.length);
      for (let i = 0; i < tok.length; i++) bytes[i] = this.u2b.get(tok.charCodeAt(i));
      return dec.decode(bytes, { stream: true });
    };
  }

  vocabSize() { return this.idToTok.length; }

  specialTokenId(text) { return this.special.get(text); }
}

// ---------------------------------------------------------------------------
// CPU sampling over GPU top-k candidates (temperature, top-k, top-p,
// optional presence penalty against a set of recently generated token ids).
export function sample({ ids, vals }, { temperature = 0.6, topK = 20, topP = 0.95, presencePenalty = 0, recentIds = null } = {}) {
  const n = ids.length / 2;
  const cand = [];
  for (let i = 0; i < n; i++) {
    const id = ids[i * 2];
    let v = vals[i * 2 + 1];
    if (presencePenalty > 0 && recentIds && recentIds.has(id)) v -= presencePenalty;
    cand.push([id, v]);
  }
  cand.sort((a, b) => b[1] - a[1]);
  const k = Math.min(topK, cand.length);
  if (temperature <= 0) return cand[0][0];
  const mx = cand[0][1];
  let probs = [], sum = 0;
  for (let i = 0; i < k; i++) {
    const p = Math.exp((cand[i][1] - mx) / temperature);
    probs.push(p); sum += p;
  }
  // top-p over the k candidates
  let cum = 0, cut = k;
  for (let i = 0; i < k; i++) {
    cum += probs[i] / sum;
    if (cum >= topP) { cut = i + 1; break; }
  }
  let r = Math.random() * probs.slice(0, cut).reduce((a, b) => a + b, 0);
  for (let i = 0; i < cut; i++) { r -= probs[i]; if (r <= 0) return cand[i][0]; }
  return cand[cut - 1][0];
}
