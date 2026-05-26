"""CLI: parse a script file and print the structured output.

Usage:
    poetry run python scripts/test_parse.py path/to/script.txt
"""

import json
import sys
from pathlib import Path

from voice_engine.parsers.script import parse_script


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_parse.py <script_path>", file=sys.stderr)
        return 1

    text = Path(sys.argv[1]).read_text(encoding="utf-8")
    lines, warnings = parse_script(text)

    print(json.dumps(
        {
            "total_lines": len(lines),
            "warnings": warnings,
            "lines": [line.model_dump() for line in lines],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
