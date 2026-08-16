"""Make the generated native stub resolve Resin's Python runner protocol."""

from pathlib import Path

STUB_PATH = Path(__file__).resolve().parents[1] / "resin" / "_resin.pyi"
IMPORT = "from resin.protocols import ConversationCodec, Runner"
OLD_IMPORT = "from resin.protocols import Runner"
GENERATED_ANNOTATION = 'runner: "resin.protocols.Runner"'
GENERATED_ANY_ANNOTATION = "runner: Any,"
GENERATED_CANCEL = "def cancel(self, /, request: Any) -> bool: ..."
GENERATED_SAMPLING_PARAMS = "def sampling_params(self, /) -> tuple:"
GENERATED_CODEC = "codec: Any,"
GENERATED_CACHE_GROUPS = "def cache_groups(self, /) -> tuple:"
GENERATED_LAYOUT_GROUPS = "def groups(self, /) -> tuple:"
GENERATED_RETENTION_DEFAULT = "retention: CacheRetention = CacheRetention.HISTORY"


def main() -> None:
    text = STUB_PATH.read_text()
    if OLD_IMPORT in text:
        text = text.replace(OLD_IMPORT, IMPORT, 1)
    elif IMPORT not in text:
        text = text.replace(
            "from typing import Any, Final, final\n",
            f"from typing import Any, Final, final\n\n{IMPORT}\n",
            1,
        )
    text = text.replace(GENERATED_ANNOTATION, "runner: Runner")
    text = text.replace(GENERATED_ANY_ANNOTATION, "runner: Runner,")
    text = text.replace(GENERATED_CODEC, "codec: ConversationCodec,")
    text = text.replace(
        GENERATED_CANCEL,
        "def cancel(self, /, request: RequestLease) -> bool: ...",
    )
    text = text.replace(
        GENERATED_SAMPLING_PARAMS,
        "def sampling_params(self, /) -> tuple[SamplingParams, ...]:",
    )
    text = text.replace(
        GENERATED_CACHE_GROUPS,
        "def cache_groups(self, /) -> tuple[CacheGroupPlan, ...]:",
    )
    text = text.replace(
        GENERATED_LAYOUT_GROUPS,
        "def groups(self, /) -> tuple[CacheGroupLayout, ...]:",
    )
    text = text.replace(
        GENERATED_RETENTION_DEFAULT,
        "retention: CacheRetention = ...",
    )
    STUB_PATH.write_text(text)


if __name__ == "__main__":
    main()
