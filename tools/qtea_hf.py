#!/usr/bin/env python3
"""CLI for Azrael QTEA Hugging Face tooling: audit, pack, verify, publish, benchmark.

Integrates the QTEA Hugging Face toolchain into a unified operational entrypoint.
All underlying functionality is implemented in the importable `orn_g4.qtea_hf` library.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path for out-of-tree execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orn_g4.errors import OrnG4Error
from orn_g4.qtea_hf import (
    DEFAULT_REPO,
    DEFAULT_SHARD_MIB,
    DEFAULT_SOURCE_BRANCH,
    DEFAULT_STAGING_BRANCH,
    audit_remote_qtea,
    pack_qtea_shards,
    publish_qtea_sharded,
    run_qtea_benchmark,
    verify_qtea_shards,
)


def cmd_audit(args: argparse.Namespace) -> int:
    report = audit_remote_qtea(
        repo=args.repo,
        revision=args.revision,
        token=args.token,
    )
    print(json.dumps(report, indent=2))
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    report = pack_qtea_shards(
        src=args.source,
        out=args.output,
        shard_mib=args.shard_mib,
        repo=args.repo,
    )
    print(json.dumps(report, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    report = verify_qtea_shards(
        out=args.path,
        source_manifest=args.source_manifest,
    )
    print(json.dumps(report, indent=2))
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    report = publish_qtea_sharded(
        repo=args.repo,
        source_revision=args.source_revision,
        branch=args.branch,
        workdir=args.workdir,
        shard_mib=args.shard_mib,
        download_workers=args.download_workers,
        resume=args.resume,
        token=args.token,
    )
    print(json.dumps(report, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    report = run_qtea_benchmark(
        source=args.source,
        repo=args.repo,
        revision=args.revision,
        cache_dir=args.cache_dir,
        device=args.device,
        prompt=args.prompt,
        max_new_tokens=args.max_new,
        output=args.output,
        token=args.token,
    )
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qtea_hf",
        description="First-party CLI for Azrael QTEA Hugging Face audit, packaging, and benchmarking.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. audit
    p_audit = sub.add_parser("audit", help="Inspect remote Hugging Face repository without downloading weights")
    p_audit.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face repository ID (default: {DEFAULT_REPO})")
    p_audit.add_argument("--revision", default=DEFAULT_SOURCE_BRANCH, help=f"Git revision/branch (default: {DEFAULT_SOURCE_BRANCH})")
    p_audit.add_argument("--token", default=None, help="Optional Hugging Face authentication token")
    p_audit.set_defaults(handler=cmd_audit)

    # 2. pack
    p_pack = sub.add_parser("pack", help="Losslessly repack local QTEA projections into indexed SafeTensors shards")
    p_pack.add_argument("--source", required=True, help="Source directory containing qtea-v1 layout")
    p_pack.add_argument("--output", required=True, help="Destination directory for sharded package")
    p_pack.add_argument("--shard-mib", type=int, default=DEFAULT_SHARD_MIB, help=f"Target shard size in MiB (default: {DEFAULT_SHARD_MIB})")
    p_pack.add_argument("--repo", default=DEFAULT_REPO, help=f"Repository identifier for metadata (default: {DEFAULT_REPO})")
    p_pack.set_defaults(handler=cmd_pack)

    # 3. verify
    p_verify = sub.add_parser("verify", help="Verify integrity of local sharded QTEA package")
    p_verify.add_argument("--path", "--source", dest="path", required=True, help="Path to sharded QTEA directory")
    p_verify.add_argument("--source-manifest", default=None, help="Optional external source manifest.json to compare against")
    p_verify.set_defaults(handler=cmd_verify)

    # 4. publish
    p_publish = sub.add_parser("publish", help="Download from HF, repack locally, and upload to staging branch")
    p_publish.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face repository ID (default: {DEFAULT_REPO})")
    p_publish.add_argument("--source-revision", default=DEFAULT_SOURCE_BRANCH, help=f"Source branch (default: {DEFAULT_SOURCE_BRANCH})")
    p_publish.add_argument("--branch", default=DEFAULT_STAGING_BRANCH, help=f"Destination staging branch (default: {DEFAULT_STAGING_BRANCH})")
    p_publish.add_argument("--workdir", default=None, help="Ephemeral scratch directory (default: system tempdir / qtea-hf-work)")
    p_publish.add_argument("--shard-mib", type=int, default=DEFAULT_SHARD_MIB, help=f"Target shard size in MiB (default: {DEFAULT_SHARD_MIB})")
    p_publish.add_argument("--download-workers", type=int, default=8, help="Number of download threads")
    p_publish.add_argument("--resume", action="store_true", help="Resume existing local pack and staging branch if verified")
    p_publish.add_argument("--token", default=None, help="Hugging Face write token (required for publication)")
    p_publish.set_defaults(handler=cmd_publish)

    # 5. benchmark
    p_bench = sub.add_parser("benchmark", help="Reconstruct complete model and execute forward/generation benchmark")
    source_group = p_bench.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", "--model-path", dest="source", default=None, help="Local directory containing QTEA package")
    source_group.add_argument("--repo", default=None, help="Hugging Face model repository ID to download from")
    p_bench.add_argument("--revision", default=DEFAULT_STAGING_BRANCH, help=f"Branch/commit when downloading from repo (default: {DEFAULT_STAGING_BRANCH})")
    p_bench.add_argument("--cache-dir", default=None, help="Local directory for downloading benchmark package")
    p_bench.add_argument("--device", default="cuda:0", help="Execution device (e.g. cuda:0, cpu; default: cuda:0)")
    p_bench.add_argument("--prompt", default="Explain briefly why the sky is blue.", help="Prompt string for benchmark")
    p_bench.add_argument("--max-new", type=int, default=8, help="Number of tokens to generate (default: 8)")
    p_bench.add_argument("--output", default=None, help="Optional output JSON file path for benchmark report")
    p_bench.add_argument("--token", default=None, help="Optional Hugging Face authentication token")
    p_bench.set_defaults(handler=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except OrnG4Error as e:
        print(f"Error [{e.__class__.__name__}]: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error [{type(e).__name__}]: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
