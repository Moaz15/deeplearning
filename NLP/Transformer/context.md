# Transformer - Context

Running context for the Transformer topic. Theory first, code later.
Update this file whenever a theory note is added, extended, or a decision is made.


## 1. Goal and authoring standard

**Goal:** make this repo a gold-standard resource for learning deep learning from
**first principles** - the kind of notes someone can read start to finish and
genuinely understand, not just recognize.

This standard governs every topic in the repo, not only the Transformer.

**First principles means:** every component is introduced by the problem it solves,
never as a fact to memorize. The order is always the same:

1. What breaks without this?
2. What is the simplest idea that could fix it, and why does that simple idea fail?
3. What is the actual mechanism?
4. Why exactly this design and not the obvious alternative?

Example of the standard already met: `03_positional_encoding.txt` earns sinusoids by
first showing self-attention is order-blind, then showing plain indices (1, 2, 3)
fail on three specific counts, then showing one sine wave collides - only then does
the formula appear. Nothing is asserted before it is needed.

**Writing rules:**

- **Simple language.** Plain English, short sentences. No jargon before it is
  defined. If a term is unavoidable, define it in the line where it first appears.
- **Real-world examples.** Every concept gets a concrete anchor - a sentence being
  translated, a word whose meaning is ambiguous, a shape being reshaped. Prefer
  reusing the repo's standing examples (section 9) so the reader tracks one story
  across notes rather than learning a new setup per note.
- **Cover the technical part fully.** Exact shapes, exact dimensions, exact order of
  operations. Precision is not the same thing as heavy math.
- **Math only when it carries understanding.** Include a formula when it is the
  clearest statement of the mechanism (the attention formula, the PE formula, the
  variance argument for sqrt(d_k)). Skip derivations, proofs, and notation that
  exist only to look rigorous. A worked numeric example beats a general proof.
- **No duplication.** Each idea has exactly one home note. Other notes link to it
  by name instead of re-explaining it.
- **No filler.** No chat residue, no praise, no restating a heading as a sentence,
  no "as we discussed above". Cut anything that does not teach.
- **Plain text, no emoji.** Notes are `.txt`; keep them readable in any editor.

**Test before considering a note done:** a reader who knows Python and basic linear
algebra, but has never seen a Transformer, can read this note alone and explain both
the mechanism and *why it exists* to someone else.

---

## 2. Current phase

**Phase 1 - Theory (in progress).** Build a complete written understanding of the
architecture in `theory/*.txt` before writing any more code.

**Phase 2 - Code (not started).** Implement the architecture from scratch in
`scratch_code/model.py`, one class per concept, following the theory notes.

Rule for this topic: a component is not implemented until its theory note is complete.

---

## 3. Theory coverage

Filenames are numbered in reading order. Each note is written to be read after the
ones before it and creates the debt the next one pays.

| # | Note | Status | Covers |
|---|---|---|---|
| 01 | `01_word_token_vector.txt` | Done, extend | Text to vectors: tokenization, IDs, embedding matrix. History: one-hot, TF-IDF and their failures |
| 02 | `02_self_attention.txt` | **Done** | Static vs contextual embeddings, why not recurrence, Q/K/V, scores/scale/softmax/blend, sqrt(d_k), worked example, masking, cost, practice Q&A |
| 03 | `03_positional_encoding.txt` | **Done** | Why order is lost, three failed fixes and the six requirements they imply, why sin+cos and geometric frequencies, formula + worked table, relative position as rotation, add vs concat, sinusoidal vs learned, code notes, practice Q&A |
| 03b | `03b_modern_positional_encoding.txt` | **Done** | Post-2017 story: why sinusoids do not extrapolate, Shaw / T5 bias / ALiBi, RoPE derived from the relative-dot-product requirement, RoPE's own wall, PI / NTK / YaRN and base bumping, NoPE and the causal-mask reversal, 2D and M-RoPE, practice Q&A |
| 04 | `04_multihead_attention.txt` | **Done** | Why one distribution is too singular, h parallel heads, shapes incl. batched `(batch, h, seq_len, d_k)`, why h heads cost the same as one, why concat + W_O, what heads really learn |
| 05 | `05_add_and_norm.txt` | **Done** | Why deep stacks stop training, residuals and the 1 + F'(x) path, the shape constraint, the residual stream, scale drift, why not BatchNorm, LayerNorm formula + gamma/beta + worked example, post-norm vs pre-norm, RMSNorm, practice Q&A |
| 06 | `06_feed_forward_network.txt` | **Empty** | Position-wise FFN, 512->2048->512, why non-linearity is needed |
| 07 | `07_encoder_architecture.txt` | Needs edit | N identical layers; should become assembly only, pointing at 04/05/06 |
| 08 | `08_masked_multihead_attention.txt` | Done | Causal mask, teacher forcing, training vs inference behaviour |
| 09 | `09_cross_attention.txt` | Done | Q from decoder, K/V from encoder, why no causal mask |
| 10 | `10_decoder_architecture.txt` | Needs edit | Three sub-blocks, right-shifted input, final linear + softmax over vocab |
| 11 | `11_training.txt` | **Empty** | Cross-entropy, label smoothing, warmup schedule, dropout placement |
| 12 | `12_inference.txt` | **Empty** | Greedy/beam/top-k/top-p, KV cache, why the encoder runs once |
| 13 | `13_architecture_variants.txt` | **Empty** | Encoder-only / decoder-only / enc-dec (BERT, GPT, T5) |
| 14 | `14_efficient_attention.txt` | **Empty** | FlashAttention, sparse and linear attention, Mamba |

"Needs edit" = content is there but violates section 1 (duplication or filler). See section 5.

Notes 01-10 are enough to write `model.py`. 11-14 explain why it is built that way.

`03b` is lettered rather than numbered because it belongs beside 03 conceptually
but is not a prerequisite for the rest of the architecture - and renumbering
04-14 was not worth it. **Read it after 08**, not after 03: its NoPE section
depends on the causal mask and its RoPE-cache section on 12's KV cache. It is the
only note that goes past 2017; note 14 (efficient attention) is its natural
sequel and the two must not duplicate each other - 03b owns position, 14 owns
the cost of attention itself.

Pending removal: `important_concepts.txt` is unnumbered because it should be
dissolved. Its pipeline walkthrough duplicates note 01 and its `/sqrt(d_k)`
explanation is owned by note 02 section 5. Move its one unique piece - why
embeddings are multiplied by `sqrt(d_model)` - into note 01, then delete it.

Supporting asset: `theory/transformers.webp` (architecture diagram).

---

## 4. Gaps to close

Ordered by the reading order in section 3, which is also the order to write them.

1. **Note 06** - extracted from note 07, which currently owns it while note 10
   duplicates it.
2. **Note 11** - cross-entropy over the vocabulary, label smoothing, the warmup
   learning-rate schedule, and dropout placement. Partly closed: note 03
   section 6.2 fixes the first dropout (applied to embedding + PE). Note 11 still
   owns the remaining placements (after each sublayer, before the residual add).
3. **Note 12** - greedy vs beam vs top-k/top-p, the generation loop, the KV cache,
   when generation stops (EOS). Now load-bearing: note 03b section 5.6 argues RoPE
   won partly because it is cache-friendly, and links here for what the cache is.
4. **Note 01 extension** - subword tokenization (BPE / WordPiece / SentencePiece),
   how `vocab_size` is chosen, special tokens, plus the `sqrt(d_model)` scaling
   rescued from `important_concepts.txt`. Note 03 section 6.2 already states the
   *reason* for that scaling (embeddings ~0.02 vs PE ~1) and points at note 01 as
   its home, so note 01 must actually receive it before 03's pointer is valid.

Closed: the padding mask now has a home in note 02 section 7.

---

## 5. Cleanup backlog

Concrete violations of section 1 already identified. Fix these while editing.

**Single-home assignments** (the note that owns each idea; everything else links to it):

| Idea | Owner note | Currently also duplicated in |
|---|---|---|
| Tokenization, IDs, embedding lookup | note 01 | `important_concepts.txt` (steps 1-3) |
| FFN (formula, 512->2048->512, why expand) | note 06 (once written) | note 07 section on FFN, note 10 section 6 |
| Add & Norm, residuals, LayerNorm rationale | note 05 | note 07 sections 3-4 and note 10 section 7 still restate it - delete on rework |
| Causal mask, teacher forcing, train vs inference | note 08 | note 10 sections 4 and 10 |
| Cross-attention mechanics | note 09 | note 10 section 5 |
| Why recurrence fails | note 02 section 2 | - resolved |
| Positional encoding, relative position, add-vs-concat | note 03 | - resolved |
| RoPE, ALiBi, T5 bias, NoPE, context extension | note 03b | - resolved |
| KV cache | note 12 | note 03b section 5.6 uses it, links out, does not explain it |
| FlashAttention / sparse / linear attention | note 14 | note 02 section 8.2 sketches it |
| `sqrt(d_model)` embedding scaling | note 01 (once extended) | `important_concepts.txt`; note 03 section 6.2 states the reason and links out |

**Filler and structural fixes:**

- Note 10 line 98 - FFN written column-vector style
  (`W2 · ReLU(W1 · x + b1) + b2`) while note 07 line 56 uses row-vector style
  (`ReLU(x W1 + b1) W2 + b2`). Standardize on row-vector form - it matches how
  `nn.Linear` actually works.
- Note 10 - roughly 90 trailing blank lines. Trim.
- Emoji and checkmark bullets appear throughout notes 07, 08. Replace with plain
  dashes.
- Note 02 section 2 is 105 lines on RNNs, correct only while `RNN/` is unwritten.
  Once `RNN/theory/` covers recurrence, shrink it to its 2.5/2.6 conclusions plus a
  pointer.

Done: note 02 fully rewritten (chat residue removed, renumbered 1-10, plain ASCII
throughout, duplication against notes 03/04/08/09 removed). `theory/.DS_Store`
removed from git and added to `.gitignore`.

---

## 6. The one-page mental model

```
SOURCE                                    TARGET (shifted right)
text                                      <START> + target tokens
 -> tokenize -> IDs                        -> tokenize -> IDs
 -> InputEmbeddings x sqrt(d_model)        -> InputEmbeddings x sqrt(d_model)
 -> + PositionalEncoding                   -> + PositionalEncoding
        |                                         |
   [ENCODER x N]                            [DECODER x N]
   Multi-Head Self-Attention                Masked Multi-Head Self-Attention
   Add & Norm                               Add & Norm
   Feed Forward (512->2048->512)            Cross-Attention  <-- K,V from encoder
   Add & Norm                               Add & Norm
        |                                   Feed Forward
        |                                   Add & Norm
        +--------- K, V --------------->          |
                                            Linear (d_model -> vocab_size)
                                            Softmax -> next-token probabilities
```

Three attention flavours, one formula:

```
Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) + mask ) V
```

| Flavour | Q from | K, V from | Mask |
|---|---|---|---|
| Encoder self-attention | encoder | encoder | padding only |
| Decoder masked self-attention | decoder | decoder | causal + padding |
| Cross-attention | decoder | encoder | padding only (no causal) |

---

## 7. Notation and shapes (canonical for both notes and code)

Use these names everywhere so the theory and the code line up.

| Symbol | Meaning | Paper value |
|---|---|---|
| `d_model` | Embedding / model dimension | 512 |
| `seq_len` / `n` | Sequence length | - |
| `vocab_size` | Tokenizer vocabulary size | - |
| `h` | Number of attention heads | 8 |
| `d_k` = `d_v` | Per-head dimension = `d_model / h` | 64 |
| `d_ff` | FFN hidden dimension | 2048 |
| `N` | Stacked encoder / decoder layers | 6 |
| `pos` | Position index in the sequence, from 0 | - |
| `i` | Positional-encoding *pair* index, 0 to `d_model/2 - 1` (dims `2i`, `2i+1`) | - |
| `10000` | PE base constant; longest wavelength is `2*pi*10000` | 10000 |

Tensor shapes through the model (batch first):

```
token IDs            (batch, seq_len)
embeddings           (batch, seq_len, d_model)
PE buffer            (1, seq_len, d_model)   broadcast over batch
+ positional enc     (batch, seq_len, d_model)
Q, K, V per head     (batch, h, seq_len, d_k)
attention scores     (batch, h, seq_len, seq_len)
head output          (batch, h, seq_len, d_k)
after concat + W_O   (batch, seq_len, d_model)
final logits         (batch, seq_len, vocab_size)
```

Two scalings that are easy to confuse:

- Embeddings are **multiplied** by `sqrt(d_model)` - once, at the input layer, to
  raise embedding magnitude to match positional encodings.
- Attention scores are **divided** by `sqrt(d_k)` - every attention call, to keep
  softmax out of its saturated region.

---

## 8. Code status

`scratch_code/model.py`:

- `InputEmbeddings` - done. `nn.Embedding` lookup scaled by `sqrt(d_model)`.
- `PositionalEncoding` - stopped mid-`__init__`. The `pe` zero matrix of shape
  `(seq_len, d_model)` is allocated; the position vector, the `div_term`, the
  sin/cos fill, the batch dimension, `register_buffer`, and `forward` are missing.
  Note 03 section 8 now spells out exactly these lines and why `div_term` is
  computed in log space - write the class against it.
- Everything else - not written: `MultiHeadAttention`, `LayerNormalization`,
  `FeedForwardBlock`, `ResidualConnection`, `EncoderBlock`, `Encoder`,
  `DecoderBlock`, `Decoder`, `ProjectionLayer`, `Transformer`, and a build helper.

Coding conventions for this topic:

- From scratch. No `nn.MultiheadAttention`, no `nn.TransformerEncoderLayer`.
- Keep paper notation (`d_model`, `seq_len`, `h`, `d_ff`) rather than plain-English
  names, so the code reads against the notes.
- One class per concept, in the same order the theory notes introduce them.
- Comments explain *why*, and point at the note that derives the *why*. The code is
  the executable half of the notes, so it is held to the same teaching standard.

---

## 9. Standing examples

Reuse these across notes so the reader follows one story instead of a new setup per note.

- Polysemy / self-attention: `"money bank grows"` vs `"river bank flows"`
  (note 02 sections 1-5).
- Worked matrix example: `"apple launched phones"` vs `"apple tastes sweet"` with
  `d_model = 2`, dimensions read as food-ness and tech-ness - full X, W_Q/W_K/W_V,
  Q K^T, scale, softmax, A V walk-through (note 02 section 6). Numbers verified;
  `W_Q W_K^T` is deliberately non-symmetric so the asymmetry argument of section 3
  is visible in the score matrix.
- Order sensitivity: `"Moaz killed Lion"` vs `"Lion killed Moaz"` (note 02
  sections 2.6 / 9, note 03 section 1).
- Positional worked table: `d_model = 4`, two frequencies (divisors 1 and 100),
  positions 0-7 (note 03 section 4.2). Values verified against Python.
- Long context: a model trained on 2,048-token documents, handed a 30,000-token
  contract and asked what clause 14 says (note 03b, throughout). The paired
  numeric anchor is the wrap table - wavelengths 6 / 600 / 6,000 / 62,800 against
  a 2,048 training length, giving 341 wraps / 3.4 / one third of a turn / 3%.
  Reused in 03b sections 1.2, 6.1 and 7.3; it is the single fact the whole
  long-context half of the note turns on.
- Head ambiguity: `"the man saw a kid with a telescope"` (note 04).
- Translation / cross-attention: English to Hindi, `"I like eating ice-cream"`
  (note 09).



## 10. How to update this file

When a theory note is written or extended:

1. Flip its row in section 3 and note what it covers.
2. Remove the matching item from section 4 if the gap is closed.
3. Remove the matching row or bullet from section 5 once cleaned.
4. Add any new symbol or shape to section 7.
5. Add any new recurring example to section 9.

When code is written, update section 8 only - the class list and what is done.
