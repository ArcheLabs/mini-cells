# EPCLM-002 — Persistent State Theory

**Project:** MiniCells / Cellular Language Models  
**Document version:** 0.2 — Research Update  
**Date:** 2026-11-08  
**Status:** A theoretical research agenda with literature and experiment updates; not an implementation specification. It does not prescribe a particular Cell network, neural operator, or GPU kernel.  
**Update log (2026-11-08):** Adds research on combining SSMs, Titans Neural Memory, and Local / Sliding Window Attention; clarifies the evidence boundaries for Titans Cell portability; adds a hybrid-architecture ablation plan at approximately 10M parameters. Existing theoretical definitions, historical experiment records, and evidence levels are retained.

**Translation:** English translation of [the Chinese v0.2 source](EPCLM-002-Persistent-State-Theory-v0.2.zh-CN.md). Section numbering, mathematical definitions, reference numbering, proposals, and evidence qualifications follow the source. Literature claims and verification dates are those reported by the supplied document; this translation does not constitute a new literature audit or experimental authorization.

---

## Abstract

EPCLM-002 studies a central hypothesis:

> **Can persistent state replace historical-token attention as the primary mechanism through which a language model stores and accesses information across time?**

This is more than replacing a Transformer's KV cache with a state tensor or adding a long-term memory module to an existing CLM. The goal is to redefine how sequence models represent history, write new information, access relevant information, and continue evolving within a finite computation budget.

This document proposes an implementation-independent theoretical framework, decomposing the problem into four interdependent capabilities:

1. **State Sufficiency:** Does the current state retain the historical information needed for future prediction?
2. **State Formation and Transition:** How is new input absorbed, merged, corrected, or forgotten?
3. **State Retrieval and Routing:** How can relevant state be located without scanning the full historical token sequence or every Cell?
4. **State-based Generation:** How can autoregressive prediction use only persistent state and a bounded current working context?

The research does not assume that fixed-size state is always sufficient. A fixed-capacity state cannot losslessly retain all information from arbitrarily long histories with arbitrary content. EPCLM must instead study how to construct sufficiently informative, updatable, addressable, and computationally effective state under explicit data distributions, future tasks, memory budgets, and error tolerances.

The document distinguishes:

- **Established research findings:** Mechanisms actually proposed and evaluated in published papers or public preprints.
- **EPCLM theoretical proposals:** Abstractions, definitions, and hypotheses proposed by this project.
- **Open questions:** Issues without theoretical proof or experimental validation.

---

# 1. Research Goals and Boundaries

## 1.1 Research goal

Establish a paradigm for language-sequence computation in which a model writes information relevant to future tasks into persistent state, rather than permanently retaining and repeatedly retrieving the full historical token representations.

Ideally, prediction at time \(t\) satisfies:

\[
P_\theta(x_{t+1}\mid x_{1:t})
\approx
P_\theta(x_{t+1}\mid S_t, W_t)
\]

where:

- \(x_{1:t}\) is the input history up to the current time.
- \(S_t\) is state that persists across tokens and input segments.
- \(W_t\) is a bounded current working context, such as the current chunk, a few recent tokens, or temporary computation results.

The central requirement is that increasing history length does not require retaining the full historical token KV. Historical information should be carried primarily by \(S_t\).

## 1.2 Questions outside this stage

EPCLM-002 does not yet prescribe:

- Whether the number of Cells is fixed or dynamically changes.
- Whether Cell State consists of vectors, matrices, MLP parameters, tensor grids, or another structure.
- Whether operators must use NCA, SSMs, RNNs, graph networks, or other mechanisms.
- Whether every form of attention must be prohibited.
- A particular GPU kernel or memory layout.
- Immediate large-model language capability or throughput.

These choices should follow theoretical analysis and experiments.

## 1.3 Boundary with earlier CLMs

Earlier X3/X4 models already have token-associated hidden states, iterative updates, local causal attention, and staged KV caches. They establish some engineering properties of recurrent state computation and incremental execution, but their states remain attached to token positions, and historical information still participates in subsequent computation through KV.

The key change in EPCLM-002 is the entity representing history:

\[
\text{Token-indexed intermediate state}
\quad\longrightarrow\quad
\text{Token-independent persistent state}
\]

Earlier CLMs remain historical baselines, rather than architectural constraints the new theory must inherit.

---

# 2. A Crucial Distinction: Replacing Attention Requires More Than Adding State

## 2.1 What attention actually does

Standard causal self-attention can be written as:

\[
o_t =
\sum_{i=1}^{t}
\alpha_{t,i}v_i
\]

\[
\alpha_{t,i} =
\operatorname{softmax}_{i\leq t}
\left(
\frac{q_t k_i^\top}{\sqrt d}
\right)
\]

It performs at least four functions:

1. **Storing historical representations:** A Transformer usually retains historical keys and values at each layer in a KV cache.
2. **Content addressing:** The current query selects historical positions using keys.
3. **Information aggregation:** Multiple historical values are combined into the current output.
4. **Contextual interaction:** Token representations influence one another during computation.

Attention combines a storage representation with dynamic retrieval and aggregation; it is not a single memory module.

## 2.2 State addresses only part of the storage problem

Compressing historical KV into a vector or matrix changes the storage format, but does not by itself explain how relevant information can be found in that state.

Conversely, storing historical information in a million Cells while computing similarities and a softmax over all Cells for every new token may simply turn token attention into Cell attention.

EPCLM must study a complete mechanism:

\[
\boxed{
\text{Persistent Memory}
+
\text{Selective Retrieval}
+
\text{State Transition}
+
\text{Readout}
}
\]

The objective is to eliminate **dense dynamic retrieval over the complete set of historical tokens**, while avoiding transferring the same global retrieval cost to the Cell set.

## 2.3 Two claims of different strength

**Claim A: Historical-Attention-Free**

No attention over unbounded historical tokens/KV. Bounded local attention inside the current chunk is allowed.

**Claim B: Attention-Free State Retrieval**

State reads also avoid query–key similarity and softmax over all candidates, using sparse indexes, explicit addresses, structured routing, or other mechanisms instead.

A is EPCLM's minimum core objective. B is a stronger research objective; neither its feasibility nor its superiority to A should be assumed.

---

# 3. Formal Problem Definition

## 3.1 History, future queries, and targets

Let the input history be:

\[
H_t = X_{1:t}=(x_1,\ldots,x_t)
\]

Future tasks should extend beyond the immediately next token. Define:

- \(Q\): a future query, current generation requirement, or task condition.
- \(Y\): the prediction target, such as the next token, a subsequent segment, a factual answer, or a task result.
- \(S_t=f_\theta(H_t)\): persistent state formed from history.
- \(W_t\): a bounded short-term working context.

The desired relationship is:

\[
P(Y\mid H_t,Q)
\approx
P_\theta(Y\mid S_t,W_t,Q)
\]

The specific future query is usually unknown while processing history. State formation must preserve information useful for the target query distribution, rather than write an answer to one already-known question.

## 3.2 Causal state updates

Persistent state must evolve online:

\[
S_0 = s_{\mathrm{init}}
\]

\[
\boxed{
S_t = F_\theta(S_{t-1},x_t)
}
\]

For chunk input \(X_{t:t+c}\), one may also define:

\[
S_{t+c}=F_\theta^{(c)}(S_t,X_{t:t+c})
\]

where \(F_\theta^{(c)}\) is a state transition over an input segment.

This specifies causality rather than the internal structure of \(F_\theta\), which may include writing, routing, local communication, state updates, and maintenance.

## 3.3 Abstract cellular state

With a Cell representation, persistent state can be written as:

\[
S_t=\{C_i^t\}_{i=1}^{N_t}
\]

A Cell may be abstracted as:

\[
C_i^t=(\mathrm{id}_i,z_i^t,\mu_i^t)
\]

where:

- \(\mathrm{id}_i\) is a stable identity or address.
- \(z_i^t\) is learnable content state.
- \(\mu_i^t\) is optional metadata, such as type, relations, time, version, or lifecycle information.

Neither a fixed key per Cell nor a fixed Cell count is required. The essential condition is that Cell identity and state are not equivalent to the positional representation of a token.

---

# 4. Core Theory I: State Sufficiency

This is EPCLM's most important mathematical question.

## 4.1 Predictive sufficiency

For a given history distribution and future-task distribution, if:

\[
P(Y\mid H_t,W_t,Q)=P(Y\mid S_t,W_t,Q)
\]

then \(S_t\) is a predictively sufficient state of history \(H_t\) for target \(Y\), conditioned on \(Q\).

Practical models allow approximate sufficiency. Define task risk:

\[
\boxed{
\mathcal{R}_{\mathrm{suff}}
=
\mathbb{E}_{H_t,W_t,Q}
\left[
D_{\mathrm{KL}}
\left(
P(Y\mid H_t,W_t,Q)
\parallel
P_\theta(Y\mid S_t,W_t,Q)
\right)
\right]
}
\]

The goal is sufficiently low risk subject to a state budget.

Using the true conditional distribution \(P(Y\mid S_t,W_t,Q)\) gives:

\[
I(Y;H_t\mid S_t,W_t,Q)
=
\mathbb{E}
\left[
D_{\mathrm{KL}}
\left(
P(Y\mid H_t,W_t,Q)
\parallel
P(Y\mid S_t,W_t,Q)
\right)
\right]
\]

Thus an equivalent expression of predictive sufficiency is:

\[
\boxed{
I(Y;H_t\mid S_t,W_t,Q)=0
}
\]

Once persistent state, the allowed bounded working context, and the future query are known, inspecting earlier raw history adds no information about \(Y\). A stronger model retaining no local context can set \(W_t=\varnothing\).

This does not require reconstructing history or retaining every token. It requires preserving information useful for the target tasks.

## 4.2 Why the task distribution must be specified

The same history supports many future questions. For example:

> Alice lives in London. She moved there in 2020.

Future queries might ask:

- Where does Alice live?
- When did she move there?
- Which words occur in the sentence?
- What does “there” refer to?
- Repeat the original text verbatim.

A compressed state may answer the first two questions without supporting verbatim reconstruction.

State Sufficiency must therefore specify a future query set \(\mathcal Q\):

\[
\mathcal{R}_{\mathrm{suff}}(\mathcal Q)
=
\mathbb E_{Q\sim\mathcal Q}
D_{\mathrm{KL}}
\left(
P(Y\mid H,Q)
\parallel
P_\theta(Y\mid S,Q)
\right)
\]

Sufficiency is relative to the target task distribution, query distribution, and error tolerance.

**EPCLM should demonstrate sufficiency for predefined task families, rather than claim lossless storage of all history.**

## 4.3 Limits of fixed-capacity state

Suppose a history of length \(T\) uses a vocabulary of size \(|\mathcal V|\). Lossless, reversible storage of any possible history requires at least:

\[
B_{\mathrm{history}}
\geq T\log_2|\mathcal V|
\]

bits.

A finite-precision state encoded with \(B\) bits can represent at most \(2^B\) distinct states. For unquantized continuous state, numerical precision, distinguishable precision, or effective encoding length must be specified before discussing practical capacity. As \(T\) grows without bound, the number of possible histories exceeds the number of representable states. Lossless storage of every history becomes impossible.

This is a fundamental finite-capacity limit, rather than a flaw of a particular Cell design.

Language modeling usually does not require lossless reconstruction of all history. Statistical regularities, redundancy, and task relevance may allow long histories to be compressed into smaller predictively sufficient representations. That possibility requires experimental validation.

## 4.4 Predictive information bottleneck

State learning can be framed as budget-constrained optimization:

\[
\boxed{
\min_\theta
I(S_t;H_t)
+
\beta\,
\mathcal R_{\mathrm{suff}}
}
\]

where:

- \(I(S_t;H_t)\) is the historical information carried by state, a theoretical proxy for compression.
- \(\mathcal R_{\mathrm{suff}}\) measures predictive loss on future tasks.
- \(\beta\) trades off information retention against task performance.

Implementations need not compute mutual information directly. State bytes, Cell count, quantization precision, estimated state entropy, task loss, and capacity curves can serve as proxies.

A more direct engineering formulation is:

\[
\min_\theta
\mathcal L_{\mathrm{task}}
+
\lambda_B\,\mathrm{MemoryBytes}(S_t)
\]

with additional constraints on read and update costs.

## 4.5 Fixed and growing state require separate study

EPCLM should not assume all Cell State must have fixed capacity. At least two theoretical routes exist.

**Fixed-capacity state**

\[
\mathrm{Memory}(S_t)\leq B
\]

This provides an explicit memory bound, but requires lossy compression, selective forgetting, or state merging.

**Expandable state**

\[
B_t \leq B_0 + g(\mathrm{relevant\ information\ in}\ H_t)
\]

Capacity may grow with information that genuinely needs retention. It need not grow linearly with every token, but cannot claim unconditional constant memory.

Expandable state introduces addressing, indexing, fragmentation, lifecycle, compression, and long-term storage costs. These routes should remain separate experimental variables rather than being merged in advance.

---

# 5. Core Theory II: State Formation and Transition

A state system must process new information correctly as well as retain it.

## 5.1 Four actions in state formation

For new input \(x_t\), the system must at least decide:

1. **Create:** Whether to create a new state entity.
2. **Write:** Which existing states receive new information.
3. **Update:** How new information is integrated with existing information.
4. **Retain / Forget:** Which older information remains, decays, is compressed, or is removed.

One model or multiple modules may perform these actions. No architecture is prescribed at this stage.

## 5.2 Abstract state transition

Let \(E_t\) be an event or write request produced by current input, and \(A_t\) the selected state entities:

\[
A_t = \mathrm{Route}_\theta(E_t,S_t,\mathcal I_t)
\]

where \(\mathcal I_t\) is an optional state index.

For active Cells:

\[
z_i^{t+1}
=
U_\theta
\left(
z_i^t,
E_t,
M_i^t,
\Delta t,
\mu_i^t
\right),
\quad i\in A_t
\]

For inactive Cells:

\[
z_i^{t+1}=z_i^t
\]

Leaving inactive Cells unchanged is an execution rule, not proof of correct routing. Important information excluded from the active set may be permanently lost.

## 5.3 New, repeated, and conflicting information

State transition must handle at least:

- **Repetition:** Avoid repeatedly duplicating an existing fact without purpose.
- **Supplementation:** Add properties such as time, source, or relations.
- **Conflict or correction:** Handle changes such as Alice moving from London to Paris.

Factual memory cannot simply average two representations. Time, provenance, confidence, and version may need retention:

\[
m=(\mathrm{subject},\mathrm{predicate},\mathrm{object},
\mathrm{time},\mathrm{source},\mathrm{confidence})
\]

This is an illustrative structured representation, not a required Cell data structure. Implicit semantics, style, pragmatics, and reasoning state may need representations different from explicit factual records.

## 5.4 Stability and plasticity

A state system needs both:

- **Plasticity:** New information can change state.
- **Stability:** New information should not inadvertently damage important retained information.

This can be expressed as multi-objective optimization:

\[
\min
\mathcal L_{\mathrm{new}}
+
\lambda_{\mathrm{retain}}\mathcal L_{\mathrm{retention}}
+
\lambda_{\mathrm{interf}}\mathcal L_{\mathrm{interference}}
\]

where:

- \(\mathcal L_{\mathrm{new}}\) measures learning or prediction on new information.
- \(\mathcal L_{\mathrm{retention}}\) measures retention of earlier tasks and facts.
- \(\mathcal L_{\mathrm{interference}}\) measures damage caused by unrelated writes.

Small state changes do not necessarily indicate good stability. Genuine corrections must allow intentional changes to old state.

---

# 6. Core Theory III: State Retrieval and Routing

This is the second key question determining whether EPCLM actually moves beyond historical attention.

## 6.1 Why state alone is insufficient

Suppose one million Cells are stored in GPU memory. For each input, computing similarity to every Cell key:

\[
s_i=q^\top k_i
\]

followed by:

\[
r=\sum_{i=1}^{N}\operatorname{softmax}(s)_i z_i
\]

still performs content-addressed aggregation over all Cells. Although this is not historical-token attention, it remains global dense retrieval and may become a bottleneck at large \(N\).

EPCLM must study memory representation and memory indexing separately.

## 6.2 Three candidate read mechanisms

### A. Dense content-addressed read

Match similarities against all state.

Advantages: simple, differentiable, and easy to train.

Disadvantages: cost usually grows linearly with Cell count, retaining an attention-like lookup over all candidates.

It is a useful correctness baseline, rather than an automatic final mechanism.

### B. Indexed sparse retrieval

First locate candidates using an index, then read a small subset:

\[
C_q=\mathrm{IndexLookup}(q,\mathcal I_t),
\quad |C_q|=K\ll N
\]

Then:

\[
r_t=\mathrm{Aggregate}(C_q)
\]

Candidate indexes include:

- Explicit addresses or symbolic keys.
- Hashing or hierarchical indexes.
- Learned routers.
- Sparse graph adjacency.
- Approximate nearest-neighbor indexes.

Ideal cost may approach:

\[
O(\mathrm{IndexCost}(N)+Kd)
\]

It cannot unconditionally be described as \(O(1)\) or \(O(\log N)\). Complexity depends on index structure, update frequency, data distribution, candidate recall, and worst cases.

### C. Structured / graph traversal

State entities maintain relations or local connections. A query enters a few starting Cells and propagates along relevant relations:

\[
C_0\rightarrow C_1\rightarrow\cdots\rightarrow C_h
\]

This may outperform global similarity retrieval for entity relations, event dependencies, or hierarchical tasks. However, traversal can propagate errors, explode into many paths, or miss information. It must be compared with indexed retrieval baselines.

## 6.3 Routing correctness must be quantified

The core measure is coverage of necessary information, not merely the fraction of active Cells. Define:

\[
\mathrm{Recall@K}
=
\frac{
\text{Necessary memory items selected by routing}
}{
\text{All necessary memory items}
}
\]

Also measure:

- The relationship between Recall@K and task accuracy.
- Router latency, memory, and energy use.
- Index update cost.
- Candidate set size \(K\).
- Recovery from incorrect routing.
- Worst cases under high load and out-of-distribution input.

If routing itself scans all Cells, sparse updates do not produce system-level sparsity.

## 6.4 Must reads use attention?

No, but a selection mechanism is still necessary whenever relevant information must be selected from multiple sources. Options include explicit addresses, indexes, rules, graph traversal, learned routing, or content similarity.

If candidate Cells are ultimately aggregated with softmax weights, the operation may still mathematically belong to the attention family, over a smaller scope.

EPCLM should distinguish:

- **Eliminating attention over historical tokens.**
- **Eliminating dense attention over all Cells.**
- **Eliminating all attention operations.**

The first is required; the other two should be decided experimentally.

---

# 7. Core Theory IV: Event-driven Sparse State Evolution

## 7.1 Basic definition

Let \(\mathcal C_t\) be all state entities, with the current event activating:

\[
A_t\subseteq \mathcal C_t
\]

The sparse update ratio is:

\[
\rho_t=\frac{|A_t|}{|\mathcal C_t|}
\]

Ideally:

\[
\rho_t\ll 1
\]

Only active states undergo expensive updates; inactive states remain unchanged.

## 7.2 An event is not a token

Events may arise from:

- New external input.
- Messages between existing states.
- Follow-up processing triggered by state changes.
- Timers or lifecycle events.
- Conflicts or dependencies requiring revalidation.

Tokens can generate events, but Event should not be defined as a synonym for Token.

## 7.3 Event propagation and termination

If Cell updates generate new events, define:

\[
E_t\rightarrow A_t\rightarrow
\{z_i^{t+1}\}\rightarrow E_{t+1}^{\mathrm{internal}}
\]

The system must prevent:

- Infinite event loops.
- Duplicate-message storms.
- State oscillation.
- Lost events.
- Uncontrolled dependence on update order.

The theoretical stage should at least specify causal ordering, deduplication semantics, conflict handling, and maximum propagation boundaries.

## 7.4 System-level conditions for sparse computation

Sparse updates matter only if all relevant costs are controlled:

\[
C_{\mathrm{total}} =
C_{\mathrm{encode}}
+C_{\mathrm{route}}
+C_{\mathrm{read}}
+C_{\mathrm{communicate}}
+C_{\mathrm{update}}
+C_{\mathrm{writeback}}
+C_{\mathrm{index}}
\]

Measuring only \(C_{\mathrm{update}}\) is insufficient. Numerous small kernels, random memory accesses, or CPU–GPU synchronizations for each Cell update may prevent theoretical sparsity from translating into speed.

---

# 8. State-based Autoregressive Generation

## 8.1 Separating current input from historical state

Distinguish:

- \(S_t\): persistent state retaining history across input segments.
- \(W_t\): bounded working state for local computation in the current segment.
- \(\theta\): long-term model parameters holding general language knowledge and update rules learned during training.

These must not be conflated.

## 8.2 Generation equations

One may define:

\[
r_t=\mathrm{Read}_\theta(S_t,W_t,q_t)
\]

\[
p_\theta(x_{t+1}\mid S_t,W_t)
=
\mathrm{Softmax}
\left(
G_\theta(r_t,W_t)
\right)
\]

After generating a new token:

\[
S_{t+1}=F_\theta(S_t,W_t,x_{t+1})
\]

Short-term working state may slide or reset with chunks while persistent state continues.

A strict Historical-Attention-Free test removes historical tokens and KV, and requires subsequent prediction to rely on persistent state.

## 8.3 Local attention should not be ruled out in advance

Local attention within the current chunk may be an efficient short-term working mechanism. EPCLM's core claim concerns independence from unbounded historical-token attention.

Experiments must separately report:

- Reads across historical tokens.
- Local computation inside the current chunk.
- Persistent-state reads.

---

# 9. Training Theory: Learning to Write Useful State

Persistent state should not depend solely on hand-written rules. Writing, updating, and reading must become coordinated mechanisms through training.

## 9.1 Joint training objective

An initial abstract objective is:

\[
\mathcal L =
\mathcal L_{\mathrm{LM}}
+\lambda_q\mathcal L_{\mathrm{query}}
+\lambda_u\mathcal L_{\mathrm{update}}
+\lambda_r\mathcal L_{\mathrm{retention}}
+\lambda_s\mathcal L_{\mathrm{sparsity}}
+\lambda_b\mathcal L_{\mathrm{budget}}
\]

The terms represent:

- \(\mathcal L_{\mathrm{LM}}\): ordinary autoregressive language modeling.
- \(\mathcal L_{\mathrm{query}}\): answering future queries from state.
- \(\mathcal L_{\mathrm{update}}\): writes, factual updates, and conflict handling.
- \(\mathcal L_{\mathrm{retention}}\): retaining old information and resisting interference.
- \(\mathcal L_{\mathrm{sparsity}}\): limiting unnecessary activations and state updates.
- \(\mathcal L_{\mathrm{budget}}\): limiting capacity, routing, and read costs.

This is not a fixed final recipe. Ablations must determine which terms are necessary.

## 9.2 Training examples must include “write–isolate–read”

Ordinary next-token training may allow dependence on visible historical tokens without learning to write persistent state. Training examples should therefore contain explicit stages:

1. Provide context.
2. Require state formation.
3. Remove raw context and historical KV.
4. Provide a query.
5. Require an answer using state alone.

This directly aligns the objective with the hypothesis.

## 9.3 Restricting historical access during training

Allowing full-history attention during training while demanding state-only inference may leave the model without sufficient learned writes. Context Removal / Memory-only Query training must ensure the read stage genuinely cannot access raw history.

## 9.4 Gradient paths for state updates

Options include:

- Full backpropagation through time.
- Chunked or truncated BPTT.
- Auxiliary predictive objectives for the writer.
- Test-time optimization of memory state.
- State snapshots, replay, and distillation.

Training choices alter capacity, write cost, and long-term stability; no one choice should be assumed optimal.

---
# 10. Relationship to Existing Research

This section summarizes related work verifiable as of 2026-10-30. The mechanisms in these papers differ from EPCLM's goals; contributions and unresolved issues are distinguished below.

| Research | Established contribution | Relationship and boundaries for EPCLM |
|---|---|---|
| Mamba (2023) | Input-dependent selective state spaces; recurrent inference without historical KV | Establishes efficient state-based sequence modeling without self-attention; its state is structured recurrent state, not an explicit Cell memory network |
| Mamba-2 / SSD (ICML 2024) | Structured duality between SSMs and attention variants | State computation and attention can have mathematical equivalences or approximations; terminology alone does not establish a paradigm shift |
| RWKV-7 (2025) | More expressive dynamic state evolution, matrix state, and recurrent inference | An important language-model baseline without standard self-attention; not an explicit persistent Cell database |
| TTT (ICML 2025) | Hidden state itself is a learning model updated at test time | A precedent for learned state-writing rules; the state remains parameterized memory |
| Titans (NeurIPS 2025) | Surprise-driven neural long-term memory combined with short-term attention | Demonstrates value in combining memory timescales; does not remove attention entirely |
| MIRAS (ICLR 2026) | Unified analysis through memory structure, attentional bias, retention gates, and memory learning algorithms | Informs writing, retention, and learning objectives; “attentional bias” is not simply Transformer self-attention |
| Nested Learning / Hope (NeurIPS 2025) | Optimization across timescales and a Continuum Memory System | Supports multiple update frequencies and self-modifying memory; does not provide EPCLM's independent Cell addressing system |
| ATLAS (ICML 2026) | Optimizes test-time memory using current and past tokens, addressing fixed capacity and purely online-update limitations | Writes should account for cross-segment accumulation and higher capacity, rather than only the latest token |
| Memory Caching (ICML 2026) | Caches recurrent-state checkpoints so effective memory grows with context | Highlights fixed-state capacity limits; snapshots reintroduce growing historical storage |
| GradMem (ICML 2026) | Test-time gradient optimization of a few memory tokens, followed by raw-context removal | A direct Context Removal precedent; not a dynamic Cell network, and reads still depend on the base model |
| Tensor Memory (2026 preprint) | Fixed-capacity 3D state tensor, soft writes, local interactions, and gated recurrence; language, image, and video evaluations | Close to spatial persistent state, but attached to Transformer blocks rather than a complete state-based LM |
| TextNCA (Findings of EMNLP 2026) | Token positions as Cells, local attention and GRU updates, shared rules across test-time iterations | Related to earlier CLMs; does not establish a reliable PPL advantage from iterative sharing itself, or persistent Cell memory across inputs |
| DNC (Nature 2016) | Differentiable external read/write memory with explicit address/content addressing | Establishes learnable access to structured external memory; content lookup remains attention-like |
| Modern Hopfield Networks (ICLR 2021) | Equivalence between modern Hopfield updates and Transformer attention | Renaming attention or using associative memory may leave the retrieval mechanism unchanged |
| Growing Neural Cellular Automata (2020) | Persistent Cell state, shared local update rules, self-organization, and regeneration | A foundational cellular-dynamics paradigm; its task is morphogenesis rather than language-history memory |

## 10.1 Principal research lessons

### Mamba: State must be selective

Ordinary SSMs with fixed dynamics struggle with information-dense discrete language. Mamba's input-dependent selection decides what to propagate or forget. EPCLM faces the same question: which information should enter state, and which should be ignored?

### TTT / Titans / MIRAS: Write rules are part of memory capability

Effective capacity depends jointly on write objectives, update algorithms, forgetting, and readout, rather than tensor size alone. Defining Cell State dimensions does not make useful memory emerge automatically.

### ATLAS / Memory Caching: Fixed-capacity recall bottlenecks are real

A fixed state may be insufficient for exact recall of many independent facts. Larger state, snapshots, or external storage can help, but reintroduce capacity and retrieval costs. Capacity–accuracy–computation curves must be central results.

### GradMem: Context Removal is an essential experimental paradigm

Raw context must actually be deleted. Adding a duplicate state without removing the original cannot establish reliance on persistent state.

### TextNCA: Cellular form alone does not establish architectural advantage

Naming token positions Cells, sharing update rules, or adding iterations does not establish persistent-state memory. State must be independent of token positions and provide measurable value across segments.

---

## 10.2 Research Update, 2026-11-08: Roles of SSMs, Neural Memory, and Attention

The question of replacing historical-token attention is further separated into three independently testable mechanisms: recurrent state, test-time neural memory, and precise local interaction. These are distinct forms of information processing.

| Mechanism | More precise definition | Main advantage | Capability that cannot be assumed |
|---|---|---|---|
| SSM / Mamba State | Input-driven recurrent hidden state evolving step by step | Continual processing of long sequences using fixed-size state without unbounded historical KV | Not an exact, addressable factual database; no guarantee of lossless history recovery |
| Titans Neural Memory | Parameters of a small neural network updated through test-time gradients | Learns nonlinear historical associations and supports query-based reads | No factual exactness guarantee; the original paper does not establish direct cross-model portability of independent user Cells |
| Local / Sliding Window Attention | Explicit content interaction among a bounded set of nearby tokens | Precise local syntax, semantics, and dependencies, with window-bounded KV | Cannot directly access tokens beyond its window by itself |
| External Knowledge | Explicit facts, documents, provenance, and versions | Auditable, exact querying, updates, and revocation | Incurs indexing, retrieval, and context-fusion costs |

**Terminology:** SSM State is better described as *recurrent / fading temporal state* than a “global knowledge base.” Recoverable information depends on dimensions, dynamics, objectives, and input distributions. Titans instead provides parameterized associative memory through gradient updates. Their complementarity requires ablation evidence.

### 10.2.1 Hybrid architectures most relevant to this research

| Research | Implemented combination | Reported evidence | Boundary for EPCLM / MiniCells |
|---|---|---|---|
| Mamba-3 (ICLR 2026) [30] | More expressive discretized recurrence, complex-valued state updates, and MIMO | At 1.5B scale, SISO reports +0.6 percentage points in average downstream accuracy over Gated DeltaNet, with MIMO adding about 1.2 points; state-size experiments approach Mamba-2 PPL with roughly half the state size | A current SSM backbone baseline; no Titans or Local Attention Cell combination experiments |
| Understanding and Enhancing Mamba-Transformer Hybrids (2025 preprint) [31] | Sequential versus parallel SSM–attention fusion | Authors report sequential hybrids working better at shorter contexts and parallel hybrids at longer contexts; paraphrase-augmented continual training improves recall | Motivates separate sequential/parallel ablations; a preprint, not evidence for Mamba + Titans |
| B'MOJO (NeurIPS 2024) [20] | Exact short-term context memory, structural memory in weights, fading SSM-state memory, and long-term eidetic external storage | Outperforms compared SSMs/hybrids on associative recall; comparable LM PPL to similarly sized Transformers/SSMs up to 1.4B; reports up to about 10% faster training; trains at 8K and tests at 32K | Direct evidence for combining SSM state and external long-term memory, but the external memory is asynchronously retrieved eidetic storage rather than Titans or independently trainable knowledge Cells |
| Titans (NeurIPS 2025) [5] | MAC uses neural-memory readout as attention context; MAG runs SWA and neural memory in parallel with gated fusion; MAL places neural memory before SWA; also an attention-free LMM | Multiple variants evaluated at scales including 170M–760M; MAC/MAG/MAL outperform Samba and Gated DeltaNet-H2 in the reported settings; MAC exceeds MAG on some longer-dependency tests; standalone LMM is also tested | Direct evidence for Titans + Local Attention, not Mamba + Titans. LMM is a sequence-model candidate rather than proof of universal attention replacement |
| TNT (ICLR 2026) [21] | Global Deep Memory across large chunks plus multiple parallel Local Deep Memories, periodically reset to break training dependencies | At 150M parameters and 10B tokens, reports up to 17.37× time-to-target-loss speedup over small-chunk Titans; multiple local memories improve LM PPL; removing global memory substantially worsens PPL | Global/local refer to internal Titans/TTT-style deep-memory hierarchy, not Mamba SSM state or independently publishable, permissioned, forkable user Cells |
| Samba (ICLR 2025) [22] | Layerwise mixture of Mamba and SWA | 3.8B parameters, 3.2T tokens; reports long-context generalization from 4K training, 256K passkey retrieval, and long-input throughput advantages over a GQA Transformer | Direct evidence for SSM + Local Attention, without Titans or independent knowledge modules |
| Griffin (2024 preprint) [23] | Gated linear recurrence (RG-LRU) with local attention | Scales to 14B; authors report matching or exceeding their Transformer baselines in several settings, with length extrapolation | RG-LRU is not Mamba SSM; a related recurrent/local-attention precedent rather than Mamba + Titans |
| RAttention (2025 preprint / Apple research page) [25] | SWA plus Residual Linear Attention (RLA) for information outside the window | At 3B/12B, reports matching or exceeding full attention/larger SWA windows on several benchmarks with a 512 window, and better RULER length generalization | RLA is an additional linear recurrent path, neither Titans nor Mamba; highlights the need for an outside-window path |
| Gated Attention (NeurIPS 2025) [24] | Head-specific sigmoid gate after SDPA output | Compares 30 variants at 15B MoE and 1.7B dense scales, about 3.5T training tokens; reports better performance, stability, length extrapolation, and reduced attention sinks | Modulates output without expanding visibility; evidence is not for Gated SWA + Titans, and its gate differs from the MAG fusion gate |
| Nemotron 3 Nano (NVIDIA technical report) [26] | 52 layers: 23 Mamba-2, 6 GQA attention, and 23 MoE layers | Reports 31.6B total and about 3.2B active parameters/token, up to 1M context; a few attention layers retain global interaction | An industrial Mamba + sparse global attention + MoE example, not local attention or Titans/shareable Cells |

### 10.2.2 Boundaries when interpreting key results

**B'MOJO:** Combining fading memory in state with long-term eidetic memory in storage demonstrates compatibility between recurrent compression and exact external storage. It does not establish SSM State + Titans parameter memory.

**Titans:** MAG is directly relevant: SWA supplies precise short-term interaction, neural memory supplies learned long-term memory, and a nonlinear gate fuses them. MAG gates *between the attention and neural-memory branches*. Gated Attention instead gates *the attention output as a function of input*. These are different modules. Standalone attention-free LMM does not establish replacement of local attention on every task.

**TNT:** Global memory persists across large chunks. During training, local memory resets per shard for parallelism, followed by small-chunk fine-tuning to restore high-resolution inference. The 17.37× result is time-to-target-loss relative to small-chunk Titans, rather than a universal advantage over optimized Transformers: Gated Transformer + FlashAttention in the same table still has a shorter training time. More local memories improve PPL, but commonsense accuracy is not monotonic; removing global memory substantially worsens results.

**Samba / Griffin / Nemotron 3:** These demonstrate complementarity between recurrence and attention, with different recurrent units and attention ranges: Mamba + SWA, RG-LRU + local attention, and Mamba-2 + a few global GQA layers, respectively. They should not be collapsed into a single generic “Mamba + Attention” result.

**RAttention:** It adds an RLA path outside the window, rather than merely shrinking a window or gating SWA. It is a relevant control for how information beyond a local window enters computation.

### 10.2.3 Current evidence matrix

| Combination | Direct research available? | Representative work | Current conclusion |
|---|---|---|---|
| SSM + Local Attention | Yes | Samba; Griffin (not Mamba SSM); Nemotron 3 (a few global attention layers) | Several feasible architectures with scaling evidence |
| Titans Neural Memory + Local Attention | Yes | Titans MAG / MAL | Direct experiments; MAG is closest to the current MiniCells concept |
| SSM + exact external memory | Yes | B'MOJO | Direct evidence, but not Titans memory |
| Multiscale / Global-Local Titans Memory | Yes | TNT | Training and quality evidence for internal hierarchical modules, not portable Cells |
| Mamba State + Titans Neural Memory | No strictly direct unified experiment found in this search | — | A combination to validate, not an established result |
| Mamba State + Local Attention + Titans Neural Memory | No strictly direct three-way ablation found in this search | — | A combination hypothesis MiniCells can study |

## 10.3 Modularity and Portability of Titans Neural Memory

### 10.3.1 What existing mechanisms establish

Titans Neural Memory is a deep network updated through gradients. Online memory includes network parameters and update state such as momentum and forgetting / weight decay. In principle, memory parameters at a given time can be saved and restored in compatible inference instances.

Three claims must be distinguished:

1. **Checkpointable:** Memory parameters and required update state can be saved and restored.
2. **Forkable:** Independent branches can be copied from a checkpoint and learn separately.
3. **Portable / Shareable:** Another model or user can load the module and read and use it correctly.

Titans validates neural-memory learning/readout and architectural variants. It does not directly validate user-level Cell registries, independent permissions, a cross-user fork marketplace, or universal loading across backbones.

TNT's global/local memory counts and resolutions establish cooperation among internal multiscale structures, rather than independent distribution outside the original model.

### 10.3.2 Portability levels

| Level | Target environment | Required conditions | Current assessment |
|---|---|---|---|
| P1: Instances of the same backbone | Identical model weights, tokenizer, memory structure, and interface | Save memory weights; continued training also requires optimizer state, steps, and update hyperparameters | Best MVP target; still requires measurement |
| P2: Same architecture, different fine-tuned versions | Same architecture, different backbone weights | Verify key/query/value representation compatibility; calibrate a reader/adapter if needed | No direct Titans-specific evidence |
| P3: Different backbones | Different model families, dimensions, or representation spaces | Shared memory latent space or a trained target-side reader/adapter | An independent research problem; direct loading cannot be assumed |

Identical hidden dimensions do not imply identical semantic spaces. Titans keys, values, and queries are usually projections of backbone representations, so memory parameters may depend on the source backbone's coordinates. Shape checks alone cannot establish sharing compatibility.

### 10.3.3 Related portable-memory research

| Research | Memory form | Transfer evaluated | Difference from Titans Cells |
|---|---|---|---|
| Frozen Memory Is Not Enough / Cross-Model Memory Transfer via Target-Side Reader Adaptation (2026 preprint) [27] | Engram-style externally addressable hash memory tables | Frozen source memory with a lightweight trained target-side reader; authors report target-side adaptation as crucial | Supports external memory assets plus reader adaptation, rather than Titans online-gradient memory |
| LatentPort (2026 preprint) [28] | KV and Gated DeltaNet recurrent inference state in hybrid models | One architecture-matched Qwen3.5 4B → 9B pair; reports transfer without target-side prefix replay | Transfers runtime inference state, not Titans weights; evidence covers one direction, one model pair, and teacher-forced continuation, not a general protocol |
| Pluggable Neural Machine Translation Models via Memory-augmented Adapters (LREC-COLING 2024) [29] | Multigranular external memory from user text plus adapters | Pluggable user-specific memory in pretrained NMT, evaluated for style/domain adaptation | A related user-memory precedent, but retrieval-augmented rather than Titans parameter memory |

### 10.3.4 Suggested MiniCells Cell asset definition

Treat portable Cells as a system design goal, rather than an already established Titans capability. A Cell package supporting continued learning should record at least:

- `memory_architecture`: network structure, depth, dimensions, and update-rule version.
- `memory_weights`: current neural-memory parameters.
- `optimizer_state`: momentum, optimizer steps, and required update state; may be omitted for inference-only use if explicitly labeled.
- `io_interface`: key/query/value dimensions, normalization, and projection protocol.
- `backbone_compatibility`: tokenizer, backbone architecture/weight version, and adapter version.
- `provenance`: knowledge sources, training-data summary, contributors, and licenses.
- `checkpoint_version`: parent version, fork source, and time.
- `permissions`: read, learn, fork, export, share, and other permissions.

The first stage requires only P1: save, load, fork, and independently evolve Cells under identical backbone weights and interfaces. P2/P3 should not block the MVP.

## 10.4 Local Attention, Gated Attention, and Long-range Information Paths

### 10.4.1 Sliding Window Attention boundaries

For causal SWA with window \(W\), position \(t\) directly accesses \([t-W+1,t]\). With a fixed window, attention computation and KV scale approximately linearly with sequence length rather than using full-history quadratic attention.

SWA alone is not long-range attention: a layer cannot directly access tokens outside its window. Multiple layers can propagate information locally, but this does not provide arbitrary historical positions with direct exact addressing. Cross-window tasks require another path, such as SSM State, Titans Memory, RLA, explicit retrieval, or a few global-attention layers.

### 10.4.2 Two gates must be distinguished

**Gated Attention (Qiu et al.):**

\[
Y=\operatorname{SDPA}(Q,K,V),\qquad
Y'=Y\odot\sigma(XW_g)
\]

This controls the strength and sparsity of current attention output. Placing the gate after SWA's SDPA output is a reasonable hypothesis, but the original evidence does not directly establish Gated SWA. It does not change the mask or expose outside-window tokens.

**Titans MAG gate:** SWA and neural memory run in parallel, with a learned nonlinear gate fusing their outputs. It controls short-term/long-term branch fusion at a different location.

Two levels of gating are possible, but require ablations: Gated SWA first, then MAG, then tests of complementarity. Combining both immediately prevents attributing changes to either one.

### 10.4.3 Three candidate long-range paths

| Path | Mechanism | Suitable role | Main costs / risks |
|---|---|---|---|
| Recurrent compression | Mamba / other SSMs | Low-cost global temporal state formed from continual input | Lossy; no exact factual-recovery guarantee |
| Parameterized association | Titans Neural Memory / multiple Cells | Reusable associations learned across chunks and sessions | Online updates, forgetting, alignment, and Cell conflicts |
| Explicit outside-window path | RAttention RLA, B'MOJO storage, RAG, or a few global-attention layers | Return outside-window information to current computation | Additional state, retrieval, or attention costs; mechanisms are not equivalent |

A candidate fusion form is:

\[
Z_t=[X_{t-W:t},\;P_s(H_t^{SSM}),\;R_t^{Titans},\;R_t^{External}]
\]

Here \(P_s\) projects SSM State into bounded state tokens, \(R_t^{Titans}\) reads selected Cells, and \(R_t^{External}\) supplies externally sourced facts. Local attention fuses these bounded representations. This is a MiniCells design proposal rather than an already validated unified architecture.

### 10.4.4 Precise terminology for “long-range attention”

Sliding Window + Gate remains local attention. Working with Mamba, Titans, or RLA may give the overall model long-range information processing, but does not make SWA itself global or long-range attention. A more precise term is **Local Attention with Recurrent / Neural / Retrieval Memory Paths**.

## 10.5 Research Gaps After This Update

Existing research separately establishes:

1. SSMs combined with local or a few global-attention layers (Samba, Griffin, Nemotron 3).
2. Titans Neural Memory combined with SWA (MAG / MAL).
3. SSMs combined with asynchronously updated exact external memory (B'MOJO).
4. Global-local multiscale training of Titans / TTT-style deep memory (TNT).
5. Restricted cross-model transfer of some external memory tables and recurrent inference states (Frozen Memory, LatentPort).

This literature check did not identify a controlled study fully validating **Mamba State + Local / Gated Attention + multiple independent Titans Neural Memory Cells**, or direct evidence of Titans Cells as universally portable user-level knowledge assets across backbones.

MiniCells should therefore focus novelty on testable questions: whether the mechanisms supply complementary information; whether memory can be independently addressed, persisted, and forked; whether compatible agents can reuse the same knowledge module; and whether total costs improve upon strong baselines such as Transformer + RAG. It should not claim to be the first to merely combine Mamba, Titans, and attention.

# 11. EPCLM Hypotheses and Falsification Conditions

EPCLM-002 treats none of the following as established facts. Each must allow experimental rejection.

## H1 — Predictive State Sufficiency

Under a given task distribution and memory budget, persistent state can retain enough historical information that removing raw tokens/KV leaves predictive loss close to the full-history baseline.

**Falsification:** State models remain substantially worse on core memory tasks despite reasonable increases in state capacity, training budget, and read capability.

## H2 — Selective State Transition

New input needs to update only a small fraction of the state system while preserving necessary prediction and memory capability.

**Falsification:** Target accuracy requires activating nearly all states, or routing cost cancels the saved update cost.

## H3 — Indexed State Retrieval

Indexes or structured routing can retrieve sufficiently relevant information without scanning all Cells.

**Falsification:** Acceptable Recall@K requires candidates approaching \(N\), or index maintenance is too expensive.

## H4 — State-based Autoregressive Generation

Persistent state supports continued generation without historical tokens/KV.

**Falsification:** Context Removal destroys basic language quality, or restoring it requires continually retaining substantial token history.

## H5 — System-level Efficiency

At equal task quality or equal quality tolerance, state mechanisms reduce actual inference cost.

**Falsification:** Total writing, routing, indexing, random access, kernel launches, and data movement cost is no lower than a strong Transformer baseline.

## H6 — Cellular Organization Adds Value

State organized into Cells with independent identities and local relations outperforms a single vector/matrix or ordinary memory tokens of equal capacity.

**Falsification:** At equal memory, parameter, and computation budgets, cellular organization has no reproducible quality, scalability, or efficiency advantage.

---

# 12. Experimental Plan

Separate theoretical validation, memory-mechanism validation, and language-model validation. Do not proceed to large-scale implementation of a later level before the preceding level passes.

## Experiment 002-A — Predictive State Sufficiency

**Goal:** Test whether bounded state can compress history while retaining information needed for future tasks.

Tasks:

- Exact recall of a single fact.
- Multiple key–value bindings.
- Multiple entities and relations.
- Delayed queries at random distances.
- Long-distance compositional reasoning.
- Varying information density and interference rates.

Controls:

1. Transformer + full KV.
2. Transformer + bounded compressed memory.
3. Standard RNN/SSM state.
4. Learnable persistent-state candidates.

Key variables: state bytes, history length, fact count, interference-token count, and query types/distributions.

Key outputs:

- Task accuracy.
- State-capacity curves.
- State bytes needed for fixed accuracy.
- Performance loss after Context Removal.
- Marginal benefit of increasing capacity.

A complex Cell network is unnecessary initially. Simple representations can establish capacity lower bounds and task-difficulty baselines.

## Experiment 002-B — Write, Update and Retention

**Goal:** Test correct absorption of new information, rather than static retention after one write.

Tasks:

- Repeated writes of the same fact.
- Adding properties to existing facts.
- Correcting existing facts.
- Conflicts between old and new facts.
- Multiple information sources.
- Large amounts of unrelated interference.
- Retention after long periods without access.

Key metrics: new-write accuracy, old-information retention, conflict-resolution accuracy, erroneous overwrite rate, state growth rate, and forgetting-policy side effects on other tasks.

## Experiment 002-C — Retrieval and Sparse Routing

**Goal:** Test effective state reads without scanning all Cells.

Controls: dense scanning/content lookup, hash/explicit addresses, hierarchical indexes, learned sparse routing, and graph traversal.

Measurements:

- Recall@K.
- End-to-end query accuracy.
- Routing latency.
- Index memory.
- Index update cost.
- GPU memory access and data movement.
- The relationship between \(K/N\) and quality.

If attention remains inside the candidate set, label it “sparse-candidate attention”; do not claim complete attention freedom.

## Experiment 002-D — State-based Autoregressive Language Modeling

**Goal:** Integrate validated state mechanisms into a small language model.

Start with the existing approximately 1.9M experimental scale, rather than investing in 30M training before mechanism validation.

Baselines:

- Transformer T1: standard Transformer.
- Native CLM X3/X4: earlier mechanisms.
- New state-based candidate: persistent state.

Tests: standard language modeling, continuous prediction across chunks, generation after historical-KV removal, long-range entity/relation dependencies, generation quality after state updates, and sustained long-running operation.

## Experiment 002-E — System Efficiency and Scaling

Run only after predefined quality and correctness gates pass.

Measure:

- Prefill / state formation.
- Per-token state updates.
- State retrieval.
- Decode latency.
- End-to-end latency.
- Peak GPU memory.
- State-storage growth rate.
- GPU DRAM traffic.
- Kernel-launch count.
- CPU–GPU synchronization.
- Energy or GPU-seconds.

Report at least two fair comparisons:

1. **Matched quality:** Compare costs at similar language quality and memory accuracy.
2. **Matched compute / memory budget:** Compare quality at similar computation/storage budgets.

Theoretical FLOPs or an isolated kernel alone are insufficient.

---

## Experiment 002-F — Hybrid-Architecture Ablations at Approximately 10M Parameters

**Goal:** Separately measure incremental contributions from local attention, gated attention, SSM State, and Titans Neural Memory in small, controlled models. The objective is mechanism validation rather than general language ability at 10M scale.

### F.1 Staged experimental matrix

**Stage 1: Establish sequence-processing baselines**

| ID | Backbone | Additional memory | Purpose |
|---|---|---|---|
| F1 | Full Attention Transformer | None | Strong historical-KV baseline |
| F2 | Sliding Window Transformer | None | Measure the cost of a local window |
| F3 | Gated Sliding Window Transformer | None | Measure head-specific gating for local attention |
| F4 | Mamba-3 SISO | None | Measure pure SSM State |
| F5-S | Mamba-3 and SWA in alternating layers / sequential composition | None | A scaled-down Samba-like mechanism baseline |
| F5-P (optional) | Parallel Mamba-3 and Gated SWA, followed by branch fusion | None | Compare sequential/parallel hybrids and test advantages at longer contexts |

**Stage 2: Add Neural Memory**

| ID | Backbone | Memory | Research question |
|---|---|---|---|
| F6 | Gated SWA Transformer | Titans MAG | Reproduce the closest published Titans + Local Attention combination |
| F7 | Titans LMM-style sequence model | Titans Neural Memory | Test deep-memory capability without attention |
| F8 | Mamba-3 | Titans | Test SSM/Titans complementarity |
| F9-S | Mamba-3 + Gated SWA (sequential / alternating) | Titans | Test sequential combination of three mechanisms |
| F9-P (optional) | Parallel Mamba-3 and Gated SWA, fused with Titans memory | Titans | Test three-way parallel fusion against sequential stacking |
| F10 (optional) | SWA + Residual Linear Attention | No Titans | A competing RAttention-like outside-window path |

**Stage 3: Multiple Cells and portability**

- Create two independent Titans Cells in a stable F6 or F9 configuration.
- Compare one Cell, two independent Cells, shared read-only Cells, and independent learning after a fork.
- Export, load, and restore Cells between instances with the same backbone and weight version.
- Then test different fine-tuned backbones with reader/adapter support; cross-family transfer is not an initial gate.

### F.2 Parameter and training fairness

- Target about 10M parameters. Report backbone static parameters, attention parameters, Titans network parameters, runtime memory parameters, and optimizer state separately.
- Control total static trainable parameters, tokenizer, data, training tokens, sequence length, optimizer, precision, seeds, and hardware as closely as possible.
- For online Titans updates, separately report inference memory parameters, trainable optimizer state, and persistence costs.
- Use identical data order and evaluation sets, multiple seeds, and both quality and system costs.
- A 10M model supports mechanism/synthetic-memory studies; its performance does not establish large-model language ability or commercial serving performance.

### F.3 Required state-clearing and memory-isolation experiments

1. **Within-session continuity:** Retain KV / SWA windows, Mamba State, and Titans Memory.
2. **Across chunks:** Clear the local window; retain Mamba State and Titans Memory.
3. **Across sessions:** Clear KV / local windows and Mamba State; restore only Titans Memory.
4. **SSM ablation:** Retain Titans, clear Mamba State, and compare with the full model.
5. **Titans ablation:** Retain Mamba State, clear Titans, and compare with the full model.
6. **Clear both:** Retain only the current input window as a short-term-context baseline.
7. **Cell fork:** Copy two Cells from one checkpoint, write different knowledge, and check isolation and independent evolution.

These conditions locate information in KV, SSM State, or Titans parameters, preventing uncleared runtime state from being mistaken for persistent knowledge.

### F.4 Evaluation tasks and metrics

| Task | Main metrics | What it validates |
|---|---|---|
| Short-range language modeling | PPL / next-token loss | Basic local semantics and language modeling |
| Delayed associative recall | Exact accuracy, Recall@K | Recovery of associations outside the window |
| Long-range fact retrieval | Exact match, source accuracy | Exact factual recovery from parameterized memory |
| Cross-session recall | Accuracy after clearing KV / SSM | Independent persistent memory from Titans |
| Continual update / correction | New-write rate, old-information retention, correction accuracy | Continual learning and forgetting |
| Cell isolation / fork | Cross-Cell contamination, post-fork behavioral differences | Module independence |
| Portability | Post-load performance difference, target-side adaptation cost | Within-/cross-backbone portability |
| System cost | Recall/learn latency, state bytes, optimizer bytes, GPU time | Practical cost advantage |

Retain a Transformer + RAG control. If Titans works only on synthetic associations while RAG is better for exact facts and provenance, report those task boundaries rather than hiding them in an aggregate score.

### F.5 Decision rules

- If F6 clearly exceeds F3, study Titans + SWA before adding Mamba.
- Retain SSM + Titans only if F8 consistently improves upon F4 and Titans-only.
- Do not retain three paths if F9 does not improve the quality–cost frontier over F6 / F8.
- Pause cross-user sharing design if Titans Cells cannot isolate or restore under the same-backbone checkpoint/fork tests.
- If different-backbone migration requires substantial target-side training, describe portability as adapter-assisted transfer rather than universal plug-and-play loading.

# 13. Experimental Methods and Fairness

## 13.1 Variables to control

- Identical tokenizer.
- Identical training data and splits.
- Equal training-token counts where applicable.
- Explicit parameter counts.
- Explicit state capacity.
- Explicit training/inference precision.
- Multiple random seeds.
- Frozen evaluation code and data.
- Reproducible hardware/software environments.

## 13.2 Provide reasonable Transformer baselines

Do not deliberately restrict a Transformer's historical access. Include at least:

- A normal full-KV baseline.
- An appropriate efficient attention implementation.
- An explicit KV budget for memory-constrained comparisons.
- A learnable compressed-memory Transformer control for Context Removal, rather than comparisons only with artificially weakened models.

## 13.3 Distinguish three efficiency measures

**Computational efficiency:** FLOPs, effective updates, and operator costs.

**Storage efficiency:** State bytes, KV bytes, index size, and peak GPU memory.

**System efficiency:** Actual latency, throughput, GPU time, data movement, and kernel launches.

Theoretical sparsity does not establish hardware acceleration; smaller state does not establish faster overall inference.

---

# 14. Principal Risks and Failure Modes

| Risk | Description | Required validation |
|---|---|---|
| Information bottleneck | Compression loses information needed by future queries | Capacity curves and random-fact recall |
| Interference | New writes damage existing memory | Repeated updates and retention tests |
| Routing misses | Necessary Cells are not activated | Recall@K and out-of-distribution routing |
| Global scanning | Router still traverses every Cell | End-to-end routing complexity |
| Cell growth | State count grows without control | Long-term growth rate |
| State oscillation | Messages cause continuing update cycles | Event termination and stability |
| Readout degradation | Stored information cannot be decoded effectively | Context Removal query tests |
| Training mismatch | Training relies on full history; inference relies on state | Forced history removal during training |
| Hardware-unfriendly execution | Sparse access causes random memory traffic and small kernels | Actual GPU profiling |
| Evaluation bias | Strong performance only on artificial memory tasks | Both natural-language and synthetic evaluations |

---

# 15. Staged Decision Gates for EPCLM-002

Follow a decision sequence rather than committing to an architecture and optimizing indefinitely.

### Gate A — Theoretical feasibility

Complete:

- Explicit predictive-sufficiency definitions.
- Capacity limits for fixed/growing state.
- Causal models for state updates and reads.
- Retrieval-complexity analysis.
- Falsifiable hypotheses.

Do not enter full implementation before completion.

### Gate B — Memory mechanism established

Demonstrate:

- Defined queries remain answerable after removing historical tokens/KV.
- New information can be written.
- Conflicts and corrections are handled.
- Unrelated input does not readily destroy retained information.

### Gate C — Sparse reads established

Demonstrate:

- Routing recall meets predefined requirements.
- Full scanning is unnecessary.
- Index/routing system costs are acceptable.

### Gate D — Language modeling established

Demonstrate:

- Autoregressive language modeling.
- Reproducible comparisons with T1 and X3/X4.
- Quantifiable language quality after Context Removal.

### Gate E — Efficiency advantage established

Claim efficiency only after observing lower actual system costs at comparable quality.

---

# 16. Potential Theoretical Contributions

Success should be described more precisely than “replacing Transformers with Cells.” Potential contributions include:

1. **Predictive Persistent State:** Representing history in state sufficient for future tasks.
2. **State Formation and Evolution:** Learning writes, integration, correction, and forgetting from continuous input.
3. **Selective State Retrieval:** Access without scanning complete history.
4. **Event-driven Computation:** Computation primarily determined by necessary state changes and propagation.
5. **State-based Autoregressive Modeling:** Generation after historical tokens/KV have been removed.
6. **Capacity–Quality–Cost Characterization:** Quantifying relationships among memory capacity, predictive quality, and actual computation cost.

The sixth contribution is especially important. Even if state-based models do not outperform Transformers on all tasks, a clear characterization of their capabilities and limitations remains valuable.

---

# 17. Current Recommended Research Decisions

Based on existing research:

| Question | Current recommendation |
|---|---|
| Continue optimizing earlier X3/X4 prefill/decode as the main direction? | Pause as the main direction; retain baselines |
| Remove every form of attention? | No; first eliminate dependence on unbounded historical-token attention |
| Require fixed-capacity state? | No; study fixed and growing state separately |
| Require explicit Cells? | A core candidate, but compare against ordinary vector/matrix state |
| Immediately implement a complex event runtime? | No; establish formation, reads, and sufficiency first |
| Immediately train at 30M? | No; validate mechanisms at smaller scale |
| First experiment? | 002-A Predictive State Sufficiency |

## 17.1 Research Roadmap Update, 2026-11-08

These recommendations update priorities without overwriting earlier theoretical hypotheses or historical experiments.

| Question | Current position |
|---|---|
| Is SSM State necessary? | Optional recurrent global temporal state; must show incremental value over Titans / Local Attention |
| Is Titans an exact knowledge base? | No; it is parameterized associative memory. External Knowledge remains responsible for exact facts, provenance, and versions |
| Must attention remain? | Do not assume global attention is required; allow local attention. Titans also tests attention-free LMM, while hybrids have task-specific advantages |
| Does SWA + Gated Attention provide long-range attention? | No; gating does not expand the window. An SSM, Titans, RLA, retrieval, or limited global-attention path is needed for outside-window information |
| Are multiple Titans memories independent Cells? | No; TNT's multiscale local memories are internal modules. Independent user Cells need lifecycle, isolation, and permission design |
| Has Titans established Cell sharing? | No; first test same-backbone checkpoint/restore/fork, then cross-backbone adapters |
| Initial model scale | Approximately 10M for mechanism validation; do not extrapolate to mature LLM performance |
| First-round main experiments | Staged F1–F9 ablations, retaining RAG and full attention as strong controls |
| Unified three-mechanism architecture | Mamba State + Gated SWA + Titans is a candidate to test, not an established result |

### Extended research proposition (2026-11-08)

> **Can sequence computation, recurrent temporal state, test-time neural memory, and exact external knowledge be separated into independently testable components, then recombined so that each component contributes measurable value in long-range understanding, cross-session learning, and portable knowledge reuse?**

This proposition does not assume Mamba, Titans, or attention must be present. Experiments may favor three paths, two paths, or even one path.

## Final Research Proposition

EPCLM-002's core proposition is:

> **Can a language model replace unbounded historical-token retrieval with a learned, persistent, selectively addressable state that preserves the information required for future prediction, while reducing total memory and computation cost?**

This proposition permits mathematical analysis, experimental validation, and falsification.

**Persistent state is not inherently superior to a KV cache. EPCLM achieves an architectural advantage only when, under a defined memory budget, it retains sufficient predictive information and its total writing, routing, reading, and update cost is lower than strong baselines.**

---

# References

The following links point to original papers, official conference pages, or research institutions. Original references were compiled as of 2026-10-30; newly added references and status checks are reported as of 2026-11-08. Preprints and conference publications are labeled separately; preprints are not automatically treated as peer-reviewed findings.

1. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** Albert Gu, Tri Dao. arXiv:2312.00752, 2023.  
   https://arxiv.org/abs/2312.00752

2. **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.** Tri Dao, Albert Gu. ICML 2024.  
   https://proceedings.mlr.press/v235/dao24a.html

3. **Learning to (Learn at Test Time): RNNs with Expressive Hidden States.** Yu Sun et al. ICML 2025.  
   https://proceedings.mlr.press/v267/sun25h.html

4. **RWKV-7 “Goose” with Expressive Dynamic State Evolution.** Bo Peng et al. arXiv:2503.14456, 2025; COLM 2025.  
   https://arxiv.org/abs/2503.14456

5. **Titans: Learning to Memorize at Test Time.** Ali Behrouz, Peilin Zhong, Vahab Mirrokni. NeurIPS 2025.  
   https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html

6. **It’s All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization.** Ali Behrouz et al. ICLR 2026.  
   https://arxiv.org/abs/2504.13173

7. **Nested Learning: The Illusion of Deep Learning Architectures.** Ali Behrouz et al. NeurIPS 2025.  
   https://papers.nips.cc/paper_files/paper/2025/hash/4309616aaed8e848009bc4a7ef73b493-Abstract-Conference.html

8. **ATLAS: Learning to Optimally Memorize the Context at Test Time.** Ali Behrouz et al. ICML 2026.  
   https://proceedings.mlr.press/v306/behrouz26b.html

9. **Memory Caching: RNNs with Growing Memory.** Ali Behrouz et al. ICML 2026.  
   https://proceedings.mlr.press/v306/behrouz26a.html

10. **GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent.** Yuri Kuratov et al. ICML 2026.  
    https://proceedings.mlr.press/v306/kuratov26a.html

11. **Tensor Memory: Fixed-Size Recurrent State for Long-Horizon Transformers.** Kabir Swain et al. arXiv:2605.27686, 2026 preprint.  
    https://arxiv.org/abs/2605.27686

12. **TextNCA: Neural Cellular Automata for Language Modeling via Hierarchical Local Attention.** Avni Mittal et al. Findings of EMNLP 2026.  
    https://aclanthology.org/2026.findings-emnlp.1413/

13. **Hybrid computing using a neural network with dynamic external memory.** Alex Graves et al. Nature 538, 471–476, 2016.  
    https://doi.org/10.1038/nature20101

14. **Hopfield Networks is All You Need.** Hubert Ramsauer et al. ICLR 2021.  
    https://arxiv.org/abs/2008.02217

15. **Growing Neural Cellular Automata.** Alexander Mordvintsev et al. Distill, 2020.  
    https://distill.pub/2020/growing-ca/

16. **Compressive Transformers for Long-Range Sequence Modelling.** Jack W. Rae et al. arXiv:1911.05507, 2019.  
    https://arxiv.org/abs/1911.05507

17. **Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention.** Tsendsuren Munkhdalai et al. arXiv:2404.07143, 2024.  
    https://arxiv.org/abs/2404.07143

18. **Retentive Network: A Successor to Transformer for Large Language Models.** Yutao Sun et al. arXiv:2307.08621, 2023.  
    https://arxiv.org/abs/2307.08621

19. **Differentiable Neural Computers — Research overview.** Google DeepMind, 2016.  
    https://deepmind.google/blog/differentiable-neural-computers/


20. **B'MOJO: Hybrid State Space Realizations of Foundation Models with Eidetic and Fading Memory.** Luca Zancato et al. NeurIPS 2024.  
    https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb6281d3f2cfe2eb8b1307227d3677b3-Abstract-Conference.html

21. **TNT: Improving Chunkwise Training for Test-Time Memorization.** Zeman Li et al. ICLR 2026.  
    https://proceedings.iclr.cc/paper_files/paper/2026/hash/bdb30687f1c2255c29b11b0b45204ebe-Abstract-Conference.html

22. **Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling.** Liliang Ren et al. ICLR 2025.  
    https://proceedings.iclr.cc/paper_files/paper/2025/hash/84a7fc24ed52e8eff514c33e8ac76ea3-Abstract-Conference.html

23. **Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.** Soham De et al. arXiv:2402.19427, 2024 preprint.  
    https://arxiv.org/abs/2402.19427

24. **Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free.** Zihan Qiu et al. NeurIPS 2025.  
    https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html

25. **RAttention: Towards the Minimal Sliding Window Size in Local-Global Attention Models.** Bailin Wang, Chang Lan, Chong Wang, Ruoming Pang. arXiv:2506.15545, 2025; Apple Machine Learning Research page, September 2025.  
    https://arxiv.org/abs/2506.15545  
    https://machinelearning.apple.com/research/rattention

26. **Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning.** NVIDIA. Technical Report, 2025.  
    https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf

27. **Frozen Memory Is Not Enough: Rethinking External Memory as Extraction.** Mingyuan Li, Guangsheng Yu, Xu Wang, Shaoxiong Ji. arXiv:2608.17050, 2026 preprint (version 5, 2026-10-24).  
    https://arxiv.org/abs/2608.17050

28. **LatentPort: Beyond KV Cache — Cross-Model Transfer of Recurrent Memory in Hybrid Language Models.** Simon P. Villani. arXiv:2609.25053, 2026 preprint.  
    https://arxiv.org/abs/2609.25053

29. **Pluggable Neural Machine Translation Models via Memory-augmented Adapters.** Yuzhuang Xu et al. LREC-COLING 2024.  
    https://aclanthology.org/2024.lrec-main.1120/

30. **Mamba-3: Improved Sequence Modeling using State Space Principles.** Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu. ICLR 2026.  
    https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html  
    https://arxiv.org/abs/2603.15569

31. **Understanding and Enhancing Mamba-Transformer Hybrids for Memory Recall and Language Modeling.** Hyunji Lee et al. arXiv:2510.26912, 2025 preprint.  
    https://arxiv.org/abs/2510.26912

---

**End of document.**
