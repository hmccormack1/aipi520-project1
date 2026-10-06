"""09 Verify the delivered source and sealed results without fitting any model.

This check requires only the Python standard library and works in a fresh clone.
The archived source is audit evidence; it is never extracted or executed.
"""
import ast
import csv
import hashlib
import json
import math
import tokenize
import io
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference_results"


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_computational_ast(source, translations):
    """Return a same-interpreter AST representation with allowed text removed."""
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in translations:
                node.value = translations[node.value]
            return node

        def strip_docstring(self, node):
            self.generic_visit(node)
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
            return node

        visit_Module = strip_docstring
        visit_FunctionDef = strip_docstring
        visit_AsyncFunctionDef = strip_docstring
        visit_ClassDef = strip_docstring

    tree = Normalize().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)


def run():
    # 09.1 Authenticate the original experiment and the translated source.
    ledger = read_json(REFERENCE / "delivery_manifest.json")
    archive = REFERENCE / "audit" / "original_frozen_source.zip"
    check(sha(archive.read_bytes()) == ledger["original_source_archive_sha256"], "Source archive changed")
    for name, expected in ledger["reference_file_hashes"].items():
        check(sha((REFERENCE / name).read_bytes()) == expected, f"Reference file changed: {name}")
    frozen_path = REFERENCE / "artifacts" / "freeze_manifest.json"
    frozen = read_json(frozen_path)
    access = read_json(REFERENCE / "data" / "test_locked" / "access_started.json")
    evaluation = read_json(REFERENCE / "artifacts" / "07_test_access_audit.json")
    original_hash = sha(frozen_path.read_bytes())
    check(access["freeze_sha256"] == original_hash == evaluation["freeze_sha256"], "Broken freeze chain")
    check(datetime.fromisoformat(frozen["frozen_at_utc"]) < datetime.fromisoformat(access["started_at_utc"]),
          "Test access did not follow the freeze")
    check(datetime.fromisoformat(access["started_at_utc"]) <= datetime.fromisoformat(evaluation["evaluated_at_utc"]),
          "Evaluation precedes access")
    with zipfile.ZipFile(archive) as source_zip:
        for name, record in ledger["source_files"].items():
            original = source_zip.read(name)
            delivered = (ROOT / name).read_bytes()
            check(sha(original) == record["original_sha256"] == frozen["source_hashes"][name], name)
            check(sha(delivered) == record["english_sha256"], f"Delivered source changed: {name}")
            if name.endswith(".py"):
                # Compare both sources with this interpreter. Historical ast.dump hashes
                # remain audit metadata because their serialization can vary by Python version.
                original_ast = normalized_computational_ast(
                    original.decode("utf-8"), ledger["message_translations"]
                )
                delivered_ast = normalized_computational_ast(
                    delivered.decode("utf-8"), ledger["message_translations"]
                )
                check(original_ast == delivered_ast, f"Computational AST differs: {name}")
                text = delivered.decode("utf-8")
                for token in tokenize.generate_tokens(io.StringIO(text).readline):
                    if token.type == tokenize.COMMENT:
                        check(token.string.isascii(), f"Non-English comment: {name}")
                for node in ast.walk(ast.parse(text)):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        docstring = ast.get_docstring(node)
                        check(docstring is None or docstring.isascii(), f"Non-English docstring: {name}")
    available_artifacts = 0
    for name, expected in frozen["artifact_hashes"].items():
        path = REFERENCE / name
        if path.exists():
            check(sha(path.read_bytes()) == expected, f"Frozen artifact changed: {name}")
            available_artifacts += 1
    for name, expected in evaluation["result_hashes"].items():
        check(sha((REFERENCE / name).read_bytes()) == expected, f"Evaluation result changed: {name}")

    # 09.2 Recompute descriptive metrics from already sealed predictions only.
    with (REFERENCE / "artifacts" / "07_final_predictions_and_actuals.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (REFERENCE / "artifacts" / "07_final_metrics.csv").open(newline="") as handle:
        recorded = {row["model"]: row for row in csv.DictReader(handle)}
    print("PASS: original freeze preceded test access; source computational AST unchanged.")
    print(f"PASS: {available_artifacts} included frozen artifacts and every sealed evaluation hash verified.")
    print("Historical raw development files are omitted from Git; their original hashes remain recorded.")
    print("All reported metrics below use frozen predictions; no training or model selection occurs.")
    print("model, n, RMSE_F, MAE_F, R2, within_2C_percent, selected_before_test")
    for model in frozen["final_models"]:
        pairs = [(float(row["actual_f"]), float(row[model])) for row in rows if row["actual_f"]]
        errors = [prediction - actual for actual, prediction in pairs]
        actual_mean = sum(actual for actual, _ in pairs) / len(pairs)
        rmse = math.sqrt(sum(error ** 2 for error in errors) / len(errors))
        mae = sum(abs(error) for error in errors) / len(errors)
        r2 = 1 - sum(error ** 2 for error in errors) / sum((actual - actual_mean) ** 2 for actual, _ in pairs)
        for field, value in (("rmse_f", rmse), ("mae_f", mae), ("r2", r2)):
            check(math.isclose(value, float(recorded[model][field]), rel_tol=1e-10, abs_tol=1e-10),
                  f"Recomputed metric differs: {model} {field}")
        hit = sum(abs(error) <= 3.6 for error in errors) / len(errors) * 100
        print(f"{model}, {len(errors)}, {rmse:.6f}, {mae:.6f}, {r2:.6f}, {hit:.2f}, {model == frozen['champion']}")
    print("The +/-2 C proportion is a post-evaluation descriptive statistic, not a selection metric.")


if __name__ == "__main__":
    run()
