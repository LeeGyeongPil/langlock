# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.3] - 2026-11-11

### Fixed
- CI: the early-closed-pipe regression test is deterministic under Python 3.14 buffering (larger fixture output; no application change).

## [0.1.2] - 2026-11-11

### Fixed
- CLI exits quietly with status 141 (128 + SIGPIPE) instead of a `BrokenPipeError` traceback when stdout is closed early, e.g. `langlock export ... | head`. 141 is a new exit code that does not collide with `check` (1) or errors (2).

## [0.1.1] - 2026-11-11

### Changed
- CI: run everything on Linux runners; the core job now covers Python 3.11-3.14 and also runs `langlock demo`.
- Tests: make the random tiny-model generate test independent of the sampling path (`min_new_tokens`).

## [0.1.0] - 2026-11-10

### Added
- Byte-level BPE `tokenizer.json` reader (`load_vocab`) that restores the real bytes of every token;
  clear `UnsupportedTokenizerError` for WordPiece and SentencePiece byte-fallback vocabularies.
- Stateful script mask (`ScriptMask`): tracks half-written UTF-8 characters, blocks Han/Kana (or other
  scripts) only when a character would complete into a banned script, keeps rare Hangul alive, blocks
  orphan continuation bytes, dead-end tails, and falls back to special tokens when nothing is left.
- `Policy` with presets `ko`, `en`, `ru`, hard/soft mode, code-block and `<think>` exemptions.
- Adapters: Hugging Face `LanglockLogitsProcessor`, llama-cpp-python `make_logits_processor`.
- Exports: llama.cpp / OpenAI `logit_bias`, GBNF, regular expression.
- `detect()` for finished text; CLI `langlock demo | scan | check | export`.
- Bundled synthetic tokenizer fixture and examples.
