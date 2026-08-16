"""Add public types that the native stub generator cannot infer."""

import ast
from pathlib import Path

STUB_PATH = Path(__file__).resolve().parents[1] / "resin" / "_resin.pyi"
U32_INPUT = "Sequence[int] | npt.NDArray[np.uint32]"

PARAMETERS = {
    ("BatchPlan", "__new__"): {
        "sequence_ids": "Sequence[int] | npt.NDArray[np.uint64]",
        "context_lengths": U32_INPUT,
        "token_ids": U32_INPUT,
        "token_indptr": U32_INPUT,
        "sampling_mask": "Sequence[bool] | npt.NDArray[np.bool_] | None",
    },
    ("CacheGroupPlan", "__new__"): {
        name: U32_INPUT
        for name in (
            "block_indptr",
            "block_ids",
            "first_logical_blocks",
            "append_block_ids",
            "append_block_offsets",
        )
    },
    ("Engine", "__new__"): {"runner": "Runner"},
    ("Engine", "cancel"): {"request": "RequestLease"},
    ("_HttpServer", "__new__"): {"codec": "ConversationCodec"},
}

RETURNS = {
    ("BatchPlan", "sequence_ids"): "npt.NDArray[np.uint64]",
    **{
        (owner, name): "npt.NDArray[np.uint32]"
        for owner, names in (
            (
                "BatchPlan",
                (
                    "context_lengths",
                    "token_ids",
                    "token_indptr",
                    "positions",
                ),
            ),
            (
                "CacheGroupPlan",
                (
                    "block_indptr",
                    "block_ids",
                    "first_logical_blocks",
                    "append_block_ids",
                    "append_block_offsets",
                ),
            ),
        )
        for name in names
    },
    ("BatchPlan", "sampling_mask"): "npt.NDArray[np.bool_]",
    ("BatchPlan", "cache_groups"): "tuple[CacheGroupPlan, ...]",
    ("BatchPlan", "sampling_params"): "tuple[SamplingParams, ...]",
    ("CacheLayout", "groups"): "tuple[CacheGroupLayout, ...]",
}


class Normalizer(ast.NodeVisitor):
    def __init__(self, source: bytes) -> None:
        self.source = source
        self.line_offsets = [0]
        self.line_offsets.extend(
            index + 1 for index, byte in enumerate(source) if byte == ord("\n")
        )
        self.owner = ""
        self.edits: list[tuple[int, int, bytes]] = []
        self.found_parameters: set[tuple[str, str, str]] = set()
        self.found_returns: set[tuple[str, str]] = set()
        self.found_retention_default = False

    def replace(self, node: ast.AST, replacement: str) -> None:
        if (
            node.lineno is None
            or node.col_offset is None
            or node.end_lineno is None
            or node.end_col_offset is None
        ):
            raise ValueError("stub AST node has no source location")
        start = self.line_offsets[node.lineno - 1] + node.col_offset
        end = self.line_offsets[node.end_lineno - 1] + node.end_col_offset
        self.edits.append((start, end, replacement.encode()))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        previous, self.owner = self.owner, node.name
        self.generic_visit(node)
        self.owner = previous

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        key = (self.owner, node.name)
        replacements = PARAMETERS.get(key, {})
        arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        by_name = {argument.arg: argument for argument in arguments}
        for name, replacement in replacements.items():
            argument = by_name.get(name)
            if argument is None or argument.annotation is None:
                raise ValueError(
                    f"generated stub is missing parameter {self.owner}.{node.name}.{name}"
                )
            self.replace(argument.annotation, replacement)
            self.found_parameters.add((*key, name))
        if replacement := RETURNS.get(key):
            if node.returns is None:
                raise ValueError(
                    f"generated stub is missing return type {self.owner}.{node.name}"
                )
            self.replace(node.returns, replacement)
            self.found_returns.add(key)
        if key == ("CacheGroupLayout", "__new__"):
            defaults = dict(
                zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
            )
            retention = by_name.get("retention")
            default = defaults.get(retention)
            if default is None:
                raise ValueError(
                    "generated stub is missing CacheGroupLayout.retention default"
                )
            self.replace(default, "...")
            self.found_retention_default = True

    def validate(self) -> None:
        expected_parameters = {
            (*key, name)
            for key, replacements in PARAMETERS.items()
            for name in replacements
        }
        missing_parameters = expected_parameters - self.found_parameters
        missing_returns = RETURNS.keys() - self.found_returns
        if missing_parameters or missing_returns or not self.found_retention_default:
            raise ValueError(
                "generated stub structure changed "
                f"(parameters={sorted(missing_parameters)}, returns={sorted(missing_returns)}, "
                f"retention_default={self.found_retention_default})"
            )


def import_edits(
    tree: ast.Module, line_offsets: list[int]
) -> list[tuple[int, int, bytes]]:
    edits: list[tuple[int, int, bytes]] = []
    imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    incomplete_imports = [
        node
        for node in imports
        if isinstance(node, ast.ImportFrom) and node.module == "_typeshed"
    ]
    for node in incomplete_imports:
        if {alias.name for alias in node.names} != {"Incomplete"}:
            raise ValueError("cannot remove a mixed _typeshed import safely")
        edits.append(
            (line_offsets[node.lineno - 1], line_offsets[node.end_lineno], b"")
        )

    has_numpy = any(
        isinstance(node, ast.Import)
        and any(alias.name == "numpy" and alias.asname == "np" for alias in node.names)
        for node in imports
    )
    has_numpy_typing = any(
        isinstance(node, ast.Import)
        and any(
            alias.name == "numpy.typing" and alias.asname == "npt"
            for alias in node.names
        )
        for node in imports
    )
    protocol_names = {
        alias.name
        for node in imports
        if isinstance(node, ast.ImportFrom) and node.module == "resin.protocols"
        for alias in node.names
    }
    additions = []
    if not has_numpy:
        additions.append("import numpy as np")
    if not has_numpy_typing:
        additions.append("import numpy.typing as npt")
    missing_protocols = {"ConversationCodec", "Runner"} - protocol_names
    if missing_protocols:
        additions.append(
            f"from resin.protocols import {', '.join(sorted(missing_protocols))}"
        )
    if additions:
        remaining_imports = [node for node in imports if node not in incomplete_imports]
        if not remaining_imports:
            raise ValueError("generated stub has no import insertion point")
        insert_at = line_offsets[max(node.end_lineno for node in remaining_imports)]
        edits.append((insert_at, insert_at, ("\n".join(additions) + "\n").encode()))
    return edits


def apply_edits(source: bytes, edits: list[tuple[int, int, bytes]]) -> bytes:
    previous_start = len(source) + 1
    for start, end, replacement in sorted(edits, reverse=True):
        if end > previous_start:
            raise ValueError("overlapping stub normalization edits")
        source = source[:start] + replacement + source[end:]
        previous_start = start
    return source


def main() -> None:
    source = STUB_PATH.read_bytes()
    tree = ast.parse(source)
    normalizer = Normalizer(source)
    normalizer.visit(tree)
    normalizer.validate()
    source = apply_edits(
        source,
        normalizer.edits + import_edits(tree, normalizer.line_offsets),
    )
    normalized = ast.parse(source)
    if any(
        isinstance(node, ast.Name) and node.id == "Incomplete"
        for node in ast.walk(normalized)
    ):
        raise ValueError("generated stub contains an unhandled incomplete type")
    STUB_PATH.write_bytes(source)


if __name__ == "__main__":
    main()
