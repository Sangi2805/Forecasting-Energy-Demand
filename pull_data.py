"""Pull DVC-tracked datasets from DagsHub storage."""

import os
import subprocess
import sys

import dagshub.auth


def main() -> int:
    token = dagshub.auth.get_token(fail_if_no_token=True)
    username = os.getenv("DAGSHUB_USER_NAME", "msshakeel12")

    commands = [
        ["dvc", "remote", "modify", "origin", "--local", "auth", "basic"],
        ["dvc", "remote", "modify", "origin", "--local", "user", username],
        ["dvc", "remote", "modify", "origin", "--local", "password", token],
        ["dvc", "remote", "default", "origin"],
        ["dvc", "pull", "-v"],
    ]

    for command in commands:
        print(f"\n$ {' '.join(command[:4])}{' ...' if 'password' in command else ''}")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(
                "\nDVC pull failed. If you see 'Missing cache files', the team may "
                "need to run `dvc push` on DagsHub first."
            )
            return result.returncode

    print("\nData pull complete. Check data/processed/ for CSV files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
