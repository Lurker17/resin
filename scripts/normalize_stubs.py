"""Make the generated native stub resolve Resin's Python runner protocol."""

from pathlib import Path

STUB_PATH = Path(__file__).resolve().parents[1] / "src" / "resin" / "_resin.pyi"
IMPORT = "from resin.protocols import Runner"
GENERATED_ANNOTATION = 'runner: "resin.protocols.Runner"'
GENERATED_CANCEL = "def cancel(self, /, request: Any) -> bool: ..."


def main() -> None:
    text = STUB_PATH.read_text()
    if IMPORT not in text:
        text = text.replace(
            "from typing import Any, Final, final\n",
            f"from typing import Any, Final, final\n\n{IMPORT}\n",
            1,
        )
    text = text.replace(GENERATED_ANNOTATION, "runner: Runner")
    text = text.replace(
        GENERATED_CANCEL,
        "def cancel(self, /, request: RequestLease) -> bool: ...",
    )
    STUB_PATH.write_text(text)


if __name__ == "__main__":
    main()
