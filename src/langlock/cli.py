"""Command line: ``langlock demo | scan | check | export``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__
from .detect import detect
from .errors import LanglockError
from .export import to_gbnf, to_logit_bias, to_regex
from .policy import PRESETS, Policy
from .vocab import load_vocab


def _add_policy_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("policy")
    group.add_argument("--policy", default="ko", choices=sorted(PRESETS), help="preset (default: ko)")
    group.add_argument("--mode", choices=("hard", "soft"), help="hard = never, soft = add --bias")
    group.add_argument("--bias", type=float, help="logit bias for soft mode, a number <= 0 (default -8)")
    group.add_argument("--exempt-code-blocks", action="store_true", help="do not block inside ``` fences")
    group.add_argument("--exempt-think", action="store_true", help="do not block inside <think>...</think>")


def _policy(args: argparse.Namespace) -> Policy:
    overrides: dict[str, object] = {}
    if args.mode:
        overrides["mode"] = args.mode
    if args.bias is not None:
        overrides["bias"] = args.bias
    if args.exempt_code_blocks:
        overrides["exempt_code_blocks"] = True
    if args.exempt_think:
        overrides["exempt_think"] = True
    return Policy.preset(args.policy, **overrides)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langlock",
        description="Keep banned writing systems (for example Chinese characters) out of LLM output.",
    )
    parser.add_argument("--version", action="version", version=f"langlock {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="practice run on a bundled tiny tokenizer (no model, no network)")

    scan = sub.add_parser("scan", help="analyse a byte-level BPE tokenizer.json")
    scan.add_argument("tokenizer", help="path to tokenizer.json")
    scan.add_argument("--json", action="store_true", help="print machine-readable JSON")
    _add_policy_options(scan)

    check = sub.add_parser("check", help="find banned characters in text files (exit 1 if any)")
    check.add_argument("files", nargs="+", help="text files, or - for standard input")
    check.add_argument("--json", action="store_true", help="print machine-readable JSON")
    check.add_argument("--exit-zero", action="store_true", help="always exit 0 (for demos and reports)")
    _add_policy_options(check)

    export = sub.add_parser("export", help="write the policy as logit_bias, GBNF or regex")
    export.add_argument("tokenizer", nargs="?", help="tokenizer.json (needed for the logit-bias formats)")
    export.add_argument(
        "--format",
        required=True,
        choices=("logit-bias", "openai-bias", "gbnf", "regex"),
        help="logit-bias = llama.cpp server list, openai-bias = {id: bias} map",
    )
    _add_policy_options(export)
    return parser


def _cmd_demo() -> int:
    from .demo import run_demo

    print(run_demo())
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    stats = load_vocab(args.tokenizer).stats(_policy(args))
    if args.json:
        print(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"vocab: {stats.vocab_size} tokens ({stats.special_tokens} special)")
    print("tokens containing a complete character of:")
    for name, count in stats.scripts.items():
        if count or name in ("Han", "Hangul"):
            print(f"  {name}: {count}")
    print(
        f"byte fragments: {stats.tail_tokens} end mid-character, "
        f"{stats.continuation_tokens} start with a continuation byte, {stats.invalid_tokens} invalid"
    )
    share = 100 * stats.static_banned / stats.vocab_size
    print(f"static banned: {stats.static_banned} ({share:.1f}% of vocab)")
    print(f"dynamic candidates: {stats.dynamic_candidates} (continuation-byte tokens checked at run time)")
    return 0


def _read(name: str) -> str:
    if name == "-":
        return sys.stdin.read()
    try:
        with open(name, encoding="utf-8") as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise LanglockError(f"cannot read {name}: {exc}") from None


def _cmd_check(args: argparse.Namespace) -> int:
    policy = _policy(args)
    failed = False
    reports = {}
    for name in args.files:
        report = detect(_read(name), policy)
        reports[name] = report
        failed = failed or not report.ok
        if args.json:
            continue
        verdict = "OK" if report.ok else "FOUND"
        print(
            f"{name}: {verdict} - {report.banned_chars} banned of {report.letters} letters "
            f"({report.ratio:.1%}), {report.replacement_chars} U+FFFD, "
            f"lines passing {report.line_pass_rate:.0%} of {report.lines}"
        )
        for line, col, script, text in report.spans:
            print(f"  {line}:{col} {script} {text}")
    if args.json:
        print(json.dumps({n: r.as_dict() for n, r in reports.items()}, ensure_ascii=False, indent=2))
    return 1 if failed and not args.exit_zero else 0


def _cmd_export(args: argparse.Namespace) -> int:
    policy = _policy(args)
    if args.format == "gbnf":
        print(to_gbnf(policy), end="")
    elif args.format == "regex":
        print(to_regex(policy))
    else:
        if not args.tokenizer:
            raise LanglockError(f"--format {args.format} needs a tokenizer.json argument")
        vocab = load_vocab(args.tokenizer)
        style = "llamacpp" if args.format == "logit-bias" else "openai"
        bias = policy.bias if policy.mode == "soft" else None
        print(json.dumps(to_logit_bias(vocab, policy, style=style, bias=bias)))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            return _cmd_demo()
        if args.command == "scan":
            return _cmd_scan(args)
        if args.command == "check":
            return _cmd_check(args)
        return _cmd_export(args)
    except LanglockError as exc:
        print(f"langlock: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
