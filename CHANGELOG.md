# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
