"""Mechanical half of the anti-slop gate. Judgement calls stay with the reviewers.

This file is exempt from its own scan, because the word list below has to contain the words it
forbids. Documents that legitimately quote the rules, such as AGENTS.md, use ALLOW_MARKER instead.
"""

import re
import subprocess
import sys

SELF = "tools/sloplint.py"
ALLOW_MARKER = "sloplint: allow"
EM_DASH = chr(0x2014)

BANNED = [
    "delve", "leverage", "seamless", "revolutionize", "revolutionise",
    "game-changing", "cutting-edge", "unlock the power", "elevate your",
    "robust solution", "in today's fast-paced world",
]

# The comma is required so that an ordinary sentence using "not just" for real emphasis
# is not caught. Only the paired construction is slop.
NOT_JUST = re.compile(r"\bnot just\b[^.;!?\n]{1,80},\s*but\b", re.IGNORECASE)

# An owner in parentheses and a reason after the colon. A bare marker rots.
TODO_OK = re.compile(r"TODO\([^)]+\):\s*\S")
TODO_ANY = re.compile(r"\bTODO\b")


def find_violations(text: str, path: str) -> list[str]:
    if path.replace("\\", "/").endswith(SELF):
        return []
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARKER in line:
            continue
        low = line.lower()
        if EM_DASH in line:
            out.append(f"{path}:{n}: em dash")
        for word in BANNED:
            if word in low:
                out.append(f"{path}:{n}: banned vocabulary: {word}")
        if NOT_JUST.search(line):
            out.append(f"{path}:{n}: paired 'not just' construction")
        if TODO_ANY.search(line) and not TODO_OK.search(line):
            out.append(f"{path}:{n}: task marker without an owner and a reason")
    return out


def main() -> int:
    files = subprocess.run(
        ["git", "ls-files", "*.py", "*.md", "*.js", "*.html", "*.css", "*.toml", "*.yml"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    problems = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            problems += find_violations(fh.read(), path)
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
