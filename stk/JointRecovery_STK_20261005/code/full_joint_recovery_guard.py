"""One small fake-clock/executor and exact-cache-key guard; no native/fit/data."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import full_joint_recovery_deadline as full


def guard():
    source = Path(full.__file__).read_text(encoding="utf-8")
    compile(source, full.__file__, "exec")  # The single explicit syntax check.
    current = [0.]
    seen, starts = [], []
    short_scope = SimpleNamespace(action_index=0, replacements=(0, 1))
    other_scope = SimpleNamespace(action_index=1, replacements=(2,))
    requests = [dict(scope=short_scope, budget_ms=10), dict(scope=short_scope, budget_ms=50),
                dict(scope=other_scope, budget_ms=200), dict(scope=other_scope, budget_ms=1000)]
    snapshot = dict(original_value_seconds=10., best_gain_seconds=10., best_value_seconds=20.,
                    original=[0], best_selected=[0], warm_by_action={"0": [0], "1": [2]}, spent_requests=[])
    costs = {"10": .1, "50": .1, "200": .3, "1000": 1.2}

    def predict(pending, state):
        current[0] += .01
        seen.append((state["best_gain_seconds"], tuple(state["warm_by_action"]["0"]), tuple(state["spent_requests"])))
        scores = []
        for request in pending:
            if request["scope"].action_index == 0:
                scores.append((12. if request["budget_ms"] == 10 else 11.) if not state["spent_requests"] else 17.)
            else:
                scores.append(13.5)
        return scores, dict(encoded_units=1, reused_units=0)

    def execute(request, state):
        current[0] += .04
        recovered = [1] if request["budget_ms"] == 10 else [0, 1]
        return dict(returned_recovery_members=recovered, diagnostics={"native_called": True})

    def admit(request, row):
        current[0] += .01
        value = 25. if request["budget_ms"] == 10 else 26.
        return dict(admitted_before_deadline=current[0] < .24, best_selected=row["returned_recovery_members"],
                    best_value_seconds=value)

    result = full.run_allocation(requests, snapshot, costs, .24, predict, execute, admit,
        lambda members: sum({0: 2., 1: 8., 2: 1.}[v] for v in members),
        lambda request, state, record: starts.append(record), clock=lambda: current[0])
    assert [r["request_key"] for r in result["actual_calls"]] == [[0, 256, 10], [0, 256, 50]]
    assert result["executed_requests"] == 2 and result["actual_native_calls"] == 2
    assert result["controller_stop_reason"] == "no_affordable_unspent_request"
    assert seen[1][0] == 15. and seen[1][1] == (1,)
    assert result["actual_calls"][0]["warm_changed"] and result["actual_calls"][1]["warm_changed"]
    assert snapshot["spent_requests"] == [] and snapshot["warm_by_action"]["0"] == [0]
    # The first selected raw prediction (12) is below the other raw value
    # (13.5): the engine actually uses marginal gain per common cost.
    assert starts[0]["predicted_gain_seconds"] == 12.
    assert len({tuple(k) for k in result["spent_requests"]}) == 2

    runtime = Path(__file__).resolve().parents[4] / "第一篇"
    cache_source = runtime / "src/joint_recovery/v4_factorized_model.py"
    node = next(n for n in ast.parse(cache_source.read_text(encoding="utf-8")).body
                if isinstance(n, ast.ClassDef) and n.name == "DecisionEmbeddingCache")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(cache_source), "exec"), namespace)
    # Exercise the actual frozen cache class without importing torch or models.
    cache = namespace["DecisionEmbeddingCache"]()
    parameter = SimpleNamespace(_version=0, device="cpu", dtype="float32")
    model = SimpleNamespace(parameters=lambda: [parameter])
    graph = object()
    cache.bind(graph, model)
    key = full.observable_unit_key((0, 1), ((0, 1),), {0}, 10.)
    cache.records[key] = "paid_encoding"
    assert full.observable_unit_key((0, 1), ((0, 1),), {0}, 10.) in cache.records
    for changed in (full.observable_unit_key((0, 1), ((0, 1),), {1}, 10.),
                    full.observable_unit_key((1, 0), ((0, 1),), {0}, 10.),
                    full.observable_unit_key((0, 1), ((0,), (1,)), {0}, 10.),
                    full.observable_unit_key((0, 1), ((0, 1),), {0}, 11.)):
        assert changed not in cache.records
    parameter._version = 1
    cache.bind(graph, model)
    assert not cache.records
    cache.records[key] = "paid_encoding"
    cache.bind(object(), model)
    assert not cache.records
    print(json.dumps(dict(status="PASS", syntax=True, fake_executor_calls=2,
                         dynamic_gstar_and_same_action_warm=True, spent_and_budget=True,
                         exact_cache_keys_and_graph_model_version=True, native_calls=0, fits=0)))


if __name__ == "__main__":
    guard()
