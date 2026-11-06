#!/usr/bin/env python3
"""Independent bounded probe; no invocation of actual C publication review."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'artifacts/provenance/C_best_publication_runbook_v1/review_metadata.py'
EXPECTED = 'a6250f88dd909567b02baa28f5d458becff169e17d5e85a37ec50f8c1e8811eb'
raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED
spec = importlib.util.spec_from_file_location('independent_C_best_authority_v2', SOURCE)
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)
reviewer.require_ROOT_batch_permission({'authorized': True})
rejected = []
for name, value in [('false', False), ('missing', ...), ('integer_one', 1),
                    ('string_true', 'true'), ('none', None)]:
    resource = {} if value is ... else {'authorized': value}
    try:
        reviewer.require_ROOT_batch_permission(resource)
    except ValueError:
        rejected.append(name)
    else:
        raise AssertionError('Nonexplicit authority accepted: ' + name)

# Confirm the newly tested function is actually reached before inherited validation.
tree = ast.parse(raw)
body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'review')
calls = [n for n in ast.walk(body) if isinstance(n, ast.Call)]
explicit = [n.lineno for n in calls if isinstance(n.func, ast.Name)
            and n.func.id == 'require_ROOT_batch_permission']
inherited = [n.lineno for n in calls if isinstance(n.func, ast.Attribute)
             and n.func.attr == 'validate_resource_review']
assert len(explicit) == len(inherited) == 1 and explicit[0] < inherited[0]

old = HERE / 'retained_a31f_review_metadata.py'
assert hashlib.sha256(old.read_bytes()).hexdigest() == 'a31f4062f53e2cfb64e028f822a30dab6e6e500918322b320716e84783f885b5'
prior = ast.parse(old.read_bytes())
old_functions = {n.name: ast.dump(n, include_attributes=False) for n in prior.body
                 if isinstance(n, ast.FunctionDef) and n.name != 'review'}
new_functions = {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name not in ('review', 'require_ROOT_batch_permission')}
assert old_functions == new_functions
old_review = next(n for n in prior.body if isinstance(n, ast.FunctionDef) and n.name == 'review')
body.body = [n for n in body.body if not (isinstance(n, ast.Expr)
    and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
    and n.value.func.id == 'require_ROOT_batch_permission')]
assert ast.dump(body, include_attributes=False) == ast.dump(old_review, include_attributes=False)

result = dict(status='passed_independent_explicit_authority_gate_and_minimal_AST_delta',
              source=dict(path=str(SOURCE), sha256=EXPECTED),
              true_accepted=True, nonexplicit_authorities_refused=rejected,
              gate_precedes_inherited_validator=True, other_existing_functions_AST_unchanged=True,
              review_AST_unchanged_except_new_explicit_gate=True,
              actual_checker_called=False, actual_authority_created=False,
              score_or_weight_payload_bytes_read=0)
with (HERE / 'probe_explicit_authority_v2_results.json').open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps(result))
