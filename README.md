# KAIRO-X

**A 42-million-parameter enforcement agent that turns any LLM into a reliable software engineer.**

KAIRO-X doesn't replace your LLM — it wraps it in an execution engine that plans tasks, implements code, verifies correctness, fixes failures, and refuses to stop until the job is actually done. The LLM decides *what* to build. KAIRO-X ensures it *gets built correctly*.

## What it does

You give KAIRO-X a task:

```
$ kairo run "Implement the full authentication system from auth_spec.md"
```

It then:

1. **Plans** — Calls the LLM to decompose the task into an execution graph of implementable components with dependency ordering.
2. **Implements** — For each component, assembles precisely the context the LLM needs (not a full repo dump — a trained neural scorer picks the right files), crafts enforcement-augmented prompts, and applies the code changes via search-and-replace edits with a 4-level fallback chain.
3. **Verifies** — Runs a two-layer verification pipeline: deterministic checks first (build, test, typecheck, lint), then an adversarial LLM audit that assumes there are bugs. If either layer fails, the agent fixes and re-verifies.
4. **Completes** — An architectural gate called the *itch register* physically prevents termination while any component remains unresolved. The agent cannot declare "done" until every node in the execution graph passes verification.

All of this runs in parallel across multiple tracks with per-file write locks, checkpoints every few steps for crash recovery, and real-time token cost tracking.

## Architecture

KAIRO-X is built on a single design principle: **rules where rules work, learning where only learning works.**

Response classification? Regex handles 80% of cases — no neural network needed. Session management? A deterministic rule table covers 90% of decisions. Template selection? A lookup table. The neural controller's 42 million parameters are focused entirely on the two problems that actually require learning:

- **Context selection** — Given a 500-file project and a task like "implement rate limiter middleware," which files should the LLM see? The right subset of types, configs, and similar patterns — not the database module, not the frontend, not the test utilities.
- **Context budget** — How many tokens to allocate per call. A simple bug fix needs 1,500 tokens. A complex module implementation needs 8,000.

Everything else is rule-based, regex-based, or lookup-table-based. This makes the system robust even before the controller is trained — the rule-based defaults produce correct behavior, and the trained controller improves efficiency.

### The seven subsystems

| Subsystem | What it does |
|---|---|
| **Execution graph** | Arena-allocated DAG with SmallVec inline edges. Tracks every component's lifecycle from pending through implementation, verification, and completion. Dynamic itch register prevents premature termination. |
| **Controller** | 12-layer liquid-hybrid recurrent network with 4 state bands (context, execution, quality, communication). GRU-style gating with cross-band multi-head attention. 6 output heads for action selection, context scoring, budget allocation, enforcement intensity, session management, and stop decisions. |
| **Context engine** | Gathers candidates from the import graph, directory structure, type definitions, test files, and dependency chain. Neural scorer ranks them. Greedy packer fills the token budget. Session-aware delta packaging avoids re-sending context the LLM already has. |
| **Verification engine** | Layer 1 (deterministic): syntax → typecheck → build → lint → targeted tests → regression tests. Layer 2 (LLM audit): adversarial prompt framing with temperature variation. Infrastructure bootstrapping sets up test frameworks for projects that lack them. |
| **Session manager** | Rule-based LLM session lifecycle. Continues sessions within the same node (the LLM just wrote this code — it's in context). Resets across nodes. Token tracking prevents context window overflow. Provider failover after 3 consecutive failures. |
| **Enforcement system** | 7 prompt templates selected by action type and priority. Language-aware placeholder detection (catches `todo!()`, `pass`, `throw new Error("not implemented")` across 5 languages). Compliance tracker escalates enforcement when the LLM repeatedly produces bad output. |
| **Parallel scheduler** | 3-track execution with per-file write locks via DashMap. Verification queue serializes deterministic checks (only one track runs `cargo test` at a time). LLM audit calls run concurrently. |

### Failure resilience

Every mechanism has a fallback:

| If this fails | Primary fix | Fallback |
|---|---|---|
| Edit search block doesn't match | Whitespace-normalized match | Fuzzy match (Levenshtein < 20%) → full file rewrite |
| LLM plan isn't parseable JSON | Re-ask LLM to fix JSON | Numbered list parser |
| LLM output truncated | Continue in same session | 3 continuations → decompose node |
| LLM asks for non-existent file | Suggest alternatives | 3 strikes → force proceed |
| API rate limited | Exponential backoff with jitter | Failover to secondary provider |
| Node fails 5 times | Restore file snapshots, mark failed | Escalate to user |
| Controller produces NaN | Reset layer state | Rule-based defaults take over |

## Project structure

```
kairo/
├── crates/
│   ├── kairo-core/          # Engine: arena, controller, context, verification, enforcement
│   │   └── src/
│   │       ├── arena/       # Execution graph with SmallVec edges, priority queue, serialization
│   │       ├── controller/  # 42M-param neural inference (liquid blocks, attention, 6 heads)
│   │       ├── context/     # Candidate gathering, import graph, neural scoring, packing
│   │       ├── verification/# Two-layer verification, infrastructure bootstrapping
│   │       ├── enforcement/ # Templates, placeholder detection, compliance tracking
│   │       ├── session/     # Session lifecycle, token tracking, cost modes
│   │       ├── parallel/    # Multi-track scheduler, verification queue
│   │       ├── tools/       # File locks, search-and-replace, filesystem, git, snapshots
│   │       ├── classify/    # Rule-based response classification with regex patterns
│   │       ├── fingerprint/ # Language, framework, build system detection
│   │       ├── persistence/ # Binary checkpointing with XXH3 integrity
│   │       ├── runtime.rs   # Main execution loop orchestrating all subsystems
│   │       └── plan_parser.rs # Two-pass plan parsing (natural language → JSON)
│   ├── kairo-cli/           # CLI: init, run, status, resume
│   └── kairo-llm/           # LLM provider abstraction (Anthropic, OpenAI, local)
├── python/
│   └── training/            # PyTorch training pipeline
│       ├── model/           # Controller architecture (liquid blocks, dense blocks, 6 heads)
│       ├── data_factory/    # Training data generation (context, budget, action, session traces)
│       ├── trainer.py       # Training loop with 3-phase curriculum
│       ├── losses.py        # 6 weighted loss functions
│       ├── curriculum.py    # Phase gating (context recall → task completion → end-to-end)
│       └── export.py        # Binary weight export for Rust inference
└── config/
    └── default_config.toml  # All configurable parameters
```

## Quick start

### Prerequisites

- Rust 1.85+ (edition 2024)
- An LLM API key (Anthropic, OpenAI, or a local server)

### Build

```bash
cd kairo
cargo build --release
```

### Initialize a project

```bash
cd your-project
kairo init
```

This scans your project and detects the language, framework, build system, test runner, linter, and type checker. Results are saved to `.kairo/config.toml`.

### Run a task

```bash
# From a description
kairo run "Add rate limiting middleware with sliding window algorithm"

# From a spec file
kairo run path/to/spec.md

# With options
kairo run "Fix the auth bug" --cost-mode efficient --parallel 1
```

### Configuration

All settings live in `.kairo/config.toml`:

```toml
[general]
cost_mode = "balanced"          # thorough | balanced | efficient
max_parallel_tracks = 3

[provider.primary]
kind = "Anthropic"
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"

[provider.fallback]
kind = "OpenAI"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
```

Three cost modes control the verification depth and token spend:

| Mode | Enforcement | Verification | Context budget |
|---|---|---|---|
| **Thorough** | Full templates, deep audits | L1 full + L2 audit + L2 deep | Generous (+30%) |
| **Balanced** | Standard templates, single audit | L1 full + L2 audit (critical/standard) | Moderate |
| **Efficient** | Minimal templates, critical-only audit | L1 full (critical gets L2) | Tight (−30%) |

## Training the controller

The controller works without training — rule-based defaults handle all decisions. Training improves context selection accuracy and token budget efficiency.

```bash
cd python

# Generate synthetic training data
python -m training.data_factory.pipeline --output data/

# Train (single GPU)
python -m training.trainer --data-dir data/ --output-dir checkpoints/

# Train (multi-GPU with DDP)
torchrun --nproc_per_node=4 -m training.trainer --distributed --data-dir data/

# Export weights for Rust inference
python -m training.export --checkpoint checkpoints/best.pt --output weights/
```

The training pipeline uses a 3-phase curriculum:
1. **Phase 1**: Context selection + budget heads only. Gate: context recall ≥ 80%.
2. **Phase 2**: Full execution traces with action selection. Gate: task completion ≥ 50%.
3. **Phase 3**: End-to-end with cost modes and session management.

## How it compares

| Capability | Claude Code | Codex | KAIRO-X |
|---|---|---|---|
| Context management | Heuristic | Sandbox-limited | Trained neural scorer |
| Task tracking | Conversation memory | Single-task | Arena DAG with itch gate |
| Verification | Self-assessment | Sandbox exec | Two-layer: deterministic + adversarial LLM |
| Completion guarantee | None | None | Architectural gate (cannot terminate while work remains) |
| Placeholder detection | None | None | Language-aware regex (5 languages) |
| Edit precision | Full file rewrites | Diffs | Search-and-replace with fuzzy fallback |
| Parallel execution | None | Possible | 3-track with file locks |
| Cost tracking | None | None | Real-time per-provider cost + token budgets |
| Crash recovery | None | None | Binary checkpoints with full state restore |

## Technical details

- **31,460 lines** of production code (25,838 Rust + 5,622 Python)
- **457 tests**, all passing
- **Zero `cargo clippy` warnings** under strict `-D warnings`
- **Arena allocation** with O(1) node access, SmallVec inline edges, FNV-hashed indices
- **Dynamic BitVec itch register** with O(1) termination check via maintained count
- **XXH3 integrity hashing** on all checkpoint data
- **Binary serialization** with <1ms checkpoint/restore for typical workloads
- **Per-file exclusive locks** via DashMap for safe parallel execution
- **Connection pooling** — cached HTTP clients across all LLM calls
- **Multi-head cross-band attention** matching Python training exactly in Rust inference

## License

MIT
