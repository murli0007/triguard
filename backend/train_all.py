from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Train models only when their real datasets are supplied. Missing datasets are reported, never replaced silently."
    )
    parser.add_argument("--url-source", choices=["uci", "uci-csv", "csv"])
    parser.add_argument("--url-csv", type=Path)
    parser.add_argument("--email-csv", type=Path)
    parser.add_argument("--malware-csv", type=Path)
    parser.add_argument("--benign-dir", type=Path)
    parser.add_argument("--malware-dir", type=Path)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Explicitly train demo artifacts for software demonstrations",
    )

    args = parser.parse_args()
    commands: list[list[str]] = []
    missing: list[str] = []

    if args.demo:
        commands = [
            ["train_url_model.py", "--source", "demo"],
            ["train_malware_model.py", "--demo"],
        ]
        if args.email_csv:
            commands.append(["train_email_model.py", "--csv", str(args.email_csv)])
        else:
            print(
                "Keeping the bundled, explicitly labeled email DEMO artifact; no synthetic data is written to real model paths."
            )
    else:
        if args.url_source == "uci":
            commands.append(["train_url_model.py", "--source", "uci"])
        elif args.url_source in {"csv", "uci-csv"} and args.url_csv:
            commands.append(
                ["train_url_model.py", "--source", args.url_source, "--csv", str(args.url_csv)]
            )
        else:
            missing.append("URL (use --url-source uci or --url-source csv/uci-csv --url-csv PATH)")

        if args.email_csv:
            commands.append(["train_email_model.py", "--csv", str(args.email_csv)])
        else:
            missing.append("email (use --email-csv PATH)")

        if args.malware_csv:
            commands.append(["train_malware_model.py", "--csv", str(args.malware_csv)])
        elif args.benign_dir and args.malware_dir:
            commands.append(
                [
                    "train_malware_model.py",
                    "--benign",
                    str(args.benign_dir),
                    "--malware",
                    str(args.malware_dir),
                ]
            )
        else:
            missing.append("malware (use --malware-csv PATH or both --benign-dir and --malware-dir)")

    for command in commands:
        print(f"\n$ python backend/{' '.join(command)}", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / command[0]), *command[1:]],
            cwd=ROOT.parent,
            check=True,
        )

    if missing:
        print("\nDatasets not supplied; these models were not retrained:")
        for item in missing:
            print(f" - {item}")

    if not commands:
        parser.print_help()


if __name__ == "__main__":
    main()
