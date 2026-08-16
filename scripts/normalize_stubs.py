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
        "sample_sequence_indices": f"{U32_INPUT} | None",
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
                    "sample_sequence_indices",
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
    ("BatchPlan", "cache_groups"): "tuple[CacheGroupPlan, ...]",
    ("BatchPlan", "sampling_params"): "tuple[SamplingParams, ...]",
    ("CacheLayout", "groups"): "tuple[CacheGroupLayout, ...]",
}


def annotation(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


class Normalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.owner = ""

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        previous, self.owner = self.owner, node.name
        self.generic_visit(node)
        self.owner = previous
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        key = (self.owner, node.name)
        replacements = PARAMETERS.get(key, {})
        arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        for argument in arguments:
            if replacement := replacements.get(argument.arg):
                argument.annotation = annotation(replacement)
        if replacement := RETURNS.get(key):
            node.returns = annotation(replacement)
        if key == ("CacheGroupLayout", "__new__"):
            index = next(
                i
                for i, argument in enumerate(node.args.kwonlyargs)
                if argument.arg == "retention"
            )
            node.args.kw_defaults[index] = ast.Constant(Ellipsis)
        return node


def main() -> None:
    tree = Normalizer().visit(ast.parse(STUB_PATH.read_text()))
    if any(
        isinstance(node, ast.Name) and node.id == "Incomplete"
        for node in ast.walk(tree)
    ):
        raise ValueError("generated stub contains an unhandled incomplete type")
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Import)
            and any(alias.name in {"numpy", "numpy.typing"} for alias in node.names)
            or isinstance(node, ast.ImportFrom)
            and node.module in {"_typeshed", "resin.protocols"}
        )
    ]
    last_import = max(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    tree.body[last_import + 1 : last_import + 1] = [
        ast.Import(names=[ast.alias("numpy", "np")]),
        ast.Import(names=[ast.alias("numpy.typing", "npt")]),
        ast.ImportFrom(
            module="resin.protocols",
            names=[ast.alias("ConversationCodec"), ast.alias("Runner")],
            level=0,
        ),
    ]
    ast.fix_missing_locations(tree)
    STUB_PATH.write_text(ast.unparse(tree) + "\n")


if __name__ == "__main__":
    main()
