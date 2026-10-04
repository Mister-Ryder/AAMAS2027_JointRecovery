"""Adapters for the official SEA-2025 CHILS and NeurIPS-2023 DIFUSCO.

No learned baseline is reimplemented here: DIFUSCO's original MISModel,
posterior, inference scheduler, and greedy decoder are called from a pinned
checkout. Its released models solve unit-weight MIS, so non-unit weights
are rejected. CHILS accepts native weighted METIS and a common incumbent.

Input NPZ: weights[n], edge_u[m], edge_v[m] (one copy of each undirected
edge). The experiment harness, not these adapters, owns graph partitions,
action menus, held-out splits, and the common feasibility/acceptance guard.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pickle
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np


OFFICIAL_COMMITS = {
    "CHILS": "515952724cd3dcc6c4365a340ecf0f1da782119a",
    "DIFUSCO": "65eb3b7bde76097c0d3438ce743fa6769baf9ce5",
}
SAT_CHECKPOINT_SHA256 = "360144ce5de7bc9dc972f0de1dc077f8c9b1abd2a7b800b833dfa32ebc5febd9"
ER_CHECKPOINT_SHA256 = "7bdfd7acee593344d323e809a99da1637a1ea2edbe7857f69957c083f21f7e0b"
DIFUSCO_SOURCE_SHA256_LF = {
    "pl_mis_model.py": "1ce4822e8c592c54afa5ba6a232931dfd1f554dff3d95b270e977b8a56efcc86",
    "pl_meta_model.py": "86790faf73a19ecb23aa02983d1298ddfbfeccc2c354431f5331c953a19d256a",
    "models/gnn_encoder.py": "2d78e25f625dcb798e0def733e90e146d86b87a72e34958f2769f0bde6c4a4b9",
    "models/nn.py": "d9547fb22b421a16dc079cb82a9d98630100b5e1b2a710d343f4d3f90b8661f3",
    "utils/diffusion_schedulers.py": "1f8c79c6425de814f3f9f9ab6760a42d363779f8c05f019f54a8bd807cef0c25",
    "utils/mis_utils.py": "c823ed0efca759945865060566d1040b28ed40d148cdef17f8de02f287213c24",
    "co_datasets/mis_dataset.py": "7db088412a445ec1ce94325e0fea76bbfb5b12a1a8d168a38e6cd520f5d8c85f",
    "utils/lr_schedulers.py": "550e5556ce2c96466a743f481bfaa00ac444742d2cdb9a43debe576b8b888238",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_graph(weights, edge_u, edge_v):
    """Canonicalize graph arrays and reject malformed inputs explicitly."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Vertex weights must be finite and strictly positive")
    u = np.asarray(edge_u, dtype=np.int64).reshape(-1)
    v = np.asarray(edge_v, dtype=np.int64).reshape(-1)
    if u.shape != v.shape:
        raise ValueError("Edge arrays must have matching lengths")
    if len(u) and ((u < 0).any() or (v < 0).any() or
                   (u >= len(weights)).any() or (v >= len(weights)).any()):
        raise ValueError("Edge endpoint lies outside the graph")
    if np.any(u == v):
        raise ValueError("Independent-set input must not contain self loops")
    pairs = sorted(set(zip(np.minimum(u, v).tolist(), np.maximum(u, v).tolist())))
    if pairs:
        pairs = np.asarray(pairs, dtype=np.int64)
        u, v = pairs[:, 0], pairs[:, 1]
    else:
        u = v = np.empty(0, dtype=np.int64)
    return weights, u, v


def validate_mask(mask, weights, edge_u, edge_v):
    mask = np.asarray(mask)
    if mask.shape != weights.shape or not np.isin(mask, (0, 1)).all():
        raise ValueError("An exact 0/1 membership vector is required")
    mask = mask.astype(np.int8)
    if np.any(mask[edge_u] & mask[edge_v]):
        raise ValueError("Baseline returned an infeasible independent set")
    return mask


def write_chils_metis(path, weights, edge_u, edge_v, weight_scale=1000):
    weights, edge_u, edge_v = validate_graph(weights, edge_u, edge_v)
    scale = int(weight_scale)
    if scale <= 0:
        raise ValueError("weight_scale must be positive")
    integer_weights = np.rint(weights * scale).astype(np.int64)
    if np.any(integer_weights <= 0):
        raise ValueError("The declared scale rounds a positive weight to zero")
    if np.sum(integer_weights, dtype=np.float64) >= np.iinfo(np.int64).max:
        raise ValueError("Scaled total weight would overflow CHILS's integer accumulator")
    neighbors = [[] for _ in weights]
    for u, v in zip(edge_u.tolist(), edge_v.tolist()):
        neighbors[u].append(v + 1)
        neighbors[v].append(u + 1)
    lines = ["%d %d 10" % (len(weights), len(edge_u))]
    lines.extend(" ".join(map(str, [int(weight)] + sorted(peers)))
                 for weight, peers in zip(integer_weights.tolist(), neighbors))
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")
    errors = np.abs(integer_weights.astype(np.float64) / scale - weights)
    return dict(weight_scale=scale, per_vertex_rounding_max=float(errors.max()) if len(errors) else 0.,
                additive_rounding_bound=float(errors.sum()), integer_objective_scale=scale)


def read_chils_solution(path, weights, edge_u, edge_v):
    values = [int(value) for value in Path(path).read_text(encoding="ascii").split()]
    if len(values) != len(set(values)) or any(v < 1 or v > len(weights) for v in values):
        raise ValueError("CHILS output must contain unique, one-indexed vertex IDs")
    mask = np.zeros(len(weights), dtype=np.int8)
    mask[np.asarray(values, dtype=np.int64) - 1] = 1
    return validate_mask(mask, weights, edge_u, edge_v)


def solve_chils(weights, edge_u, edge_v, binary, total_seconds=1., seed=17,
                initial_mask=None, population=16, threads=1, weight_scale=1000,
                alternating_seconds=None):
    """Run the official executable and retain a real feasible output.

    total_seconds includes canonicalization, export, and subprocess overhead.
    CHILS measures its own search time internally. A soft termination is sent
    at the outer deadline; its official signal handler writes the incumbent.
    Actual overshoot and any missing output are returned, never concealed as
    successful baseline execution. Validation remains part of online time.
    """
    started = time.perf_counter()
    weights, edge_u, edge_v = validate_graph(weights, edge_u, edge_v)
    if total_seconds <= 0 or population < 1 or threads < 1:
        raise ValueError("Time, population, and thread count must be positive")
    if initial_mask is None:
        initial = np.zeros(len(weights), dtype=np.int8)
    else:
        initial = validate_mask(initial_mask, weights, edge_u, edge_v)
    if not len(weights):
        return initial, dict(status=0, returned_solution=True, empty_graph=True,
                             online_ms=(time.perf_counter() - started) * 1000)
    with tempfile.TemporaryDirectory(prefix="aamas-chils-") as folder:
        folder = Path(folder)
        graph_path = folder / "instance.graph"
        warmstart_path = folder / "incumbent.txt"
        output_path = folder / "solution.txt"
        conversion = write_chils_metis(graph_path, weights, edge_u, edge_v, weight_scale)
        warmstart_path.write_text("".join("%d\n" % (v + 1) for v in np.flatnonzero(initial)),
                                  encoding="ascii")
        prep_seconds = time.perf_counter() - started
        remaining = float(total_seconds) - prep_seconds
        if remaining <= 0:
            return initial, dict(status="preparation_exceeded_deadline", returned_solution=False,
                                 online_ms=(time.perf_counter() - started) * 1000,
                                 requested_ms=total_seconds * 1000, preparation_ms=prep_seconds * 1000,
                                 deadline_exceeded=True, **conversion)
        command = [str(binary), "-g", str(graph_path), "-i", str(warmstart_path),
                   "-o", str(output_path), "-p", str(int(population)), "-c", str(int(threads)),
                   "-t", str(remaining), "-r", str(int(seed))]
        if alternating_seconds is not None:
            command.extend(["-s", str(float(alternating_seconds))])
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(threads))
        env["OMP_PROC_BIND"] = "true"
        env["OPENBLAS_NUM_THREADS"] = "1"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, env=env)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=max(.001, total_seconds -
                                                            (time.perf_counter() - started)))
        except subprocess.TimeoutExpired:
            timed_out = True
            process.send_signal(signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=.5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
        returned = output_path.exists()
        output = read_chils_solution(output_path, weights, edge_u, edge_v) if returned else initial
        raw_value = float(weights @ output)
        # Common acceptance guard: the warm start remains available even if a
        # baseline fails to return a better feasible schedule by its deadline.
        retained_incumbent = raw_value < float(weights @ initial) - 1e-9
        accepted = initial if retained_incumbent else output
        elapsed = time.perf_counter() - started
        details = dict(status=process.returncode, returned_solution=returned,
                       requested_ms=total_seconds * 1000, online_ms=elapsed * 1000,
                       preparation_ms=prep_seconds * 1000, sent_deadline_signal=timed_out,
                       deadline_exceeded=elapsed > total_seconds, raw_value=raw_value,
                       accepted_value=float(weights @ accepted), retained_incumbent=retained_incumbent,
                       seed=int(seed), population=int(population), threads=int(threads),
                       alternating_seconds=alternating_seconds,
                       stdout=stdout.strip(), stderr=stderr.strip(), **conversion)
        return accepted, details


class OfficialDIFUSCO:
    """Loaded released model; all architecture and denoising stay upstream."""

    def __init__(self, checkout, checkpoint, device="cuda", diffusion_type="categorical",
                 steps=50, sequential_samples=1, parallel_samples=1, verify_sat_hash=True):
        setup_started = time.perf_counter()
        checkout, checkpoint = Path(checkout).resolve(), Path(checkpoint).resolve()
        model_path = checkout / "difusco"
        if not (model_path / "pl_mis_model.py").is_file():
            raise ValueError("An official DIFUSCO source checkout is required")
        verified_source = {}
        for relative_path, expected in DIFUSCO_SOURCE_SHA256_LF.items():
            # Check LF-normalized source for cross-platform Git checkout parity.
            actual = hashlib.sha256((model_path / relative_path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            if actual != expected:
                raise ValueError("DIFUSCO source differs from the pinned official implementation: " + relative_path)
            verified_source[relative_path] = actual
        checksum = sha256_file(checkpoint)
        expected_checksum = SAT_CHECKPOINT_SHA256 if diffusion_type == "categorical" else ER_CHECKPOINT_SHA256
        if verify_sat_hash and checksum != expected_checksum:
            raise ValueError("Checkpoint checksum differs from the released MIS checkpoint")
        if steps < 1 or sequential_samples < 1 or parallel_samples < 1:
            raise ValueError("Diffusion steps and sample counts must be positive")
        if sequential_samples > 1 and parallel_samples > 1:
            raise ValueError("Use one sampling axis: the upstream mixed-axis test hook repeatedly duplicates edges")
        sys.path.insert(0, str(model_path))
        # Python >=3.8 has native protocol-5 pickle. This compatibility alias
        # only replaces the upstream dataset module's obsolete pickle5 import.
        sys.modules.setdefault("pickle5", pickle)
        import torch
        from pl_mis_model import MISModel
        from utils.diffusion_schedulers import InferenceSchedule
        from utils.mis_utils import mis_decode_np
        if Path(importlib.import_module("pl_mis_model").__file__).resolve().parent != model_path:
            raise RuntimeError("A conflicting pl_mis_model module shadowed the pinned checkout")
        args = SimpleNamespace(
            diffusion_type=diffusion_type, diffusion_schedule="linear", diffusion_steps=1000,
            sparse_factor=-1, n_layers=12, hidden_dim=256, aggregation="sum",
            use_activation_checkpoint=False, inference_trick="ddim",
            training_split_label_dir=None, storage_path=str(checkpoint.parent),
            training_split="__no_training_data__*.gpickle", test_split="__no_test_data__*.gpickle",
            validation_split="__no_validation_data__*.gpickle", inference_schedule="cosine",
            inference_diffusion_steps=int(steps), sequential_sampling=int(sequential_samples),
            parallel_sampling=int(parallel_samples), batch_size=1, num_workers=0,
            learning_rate=.0002, weight_decay=.0001, lr_scheduler="cosine-decay")
        self.model = MISModel(param_args=args)
        package = torch.load(str(checkpoint), map_location="cpu")
        self.model.load_state_dict(package["state_dict"], strict=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.torch = torch
        self.schedule_class = InferenceSchedule
        self.decode = mis_decode_np
        self.args = args
        self.setup = dict(checkpoint_sha256=checksum, source_commit=OFFICIAL_COMMITS["DIFUSCO"],
                          verified_source_sha256_lf=verified_source,
                          checkpoint_epoch=package.get("epoch"), checkpoint_step=package.get("global_step"),
                          released_lightning_version=package.get("pytorch-lightning_version"),
                          setup_seconds=time.perf_counter() - setup_started,
                          device=str(device), diffusion_type=diffusion_type,
                          inference_steps=int(steps), sequential_samples=int(sequential_samples),
                          parallel_samples=int(parallel_samples), actual_torch_version=torch.__version__)

    def _sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def solve(self, weights, edge_u, edge_v, seed=17, initial_mask=None):
        """Original test-step sampling, exposing membership and timing.

        Fixed inference steps/sample counts determine the work; there is no
        claim that this call obeys an arbitrary millisecond deadline. A sweep
        reports measured wall time with synchronized GPU and the actual mask.
        The official algorithm receives neither weights nor an incumbent.
        """
        self._sync()
        started = time.perf_counter()
        weights, u, v = validate_graph(weights, edge_u, edge_v)
        if not np.all(weights == 1.):
            raise ValueError("Released DIFUSCO MIS models do not support non-unit vertex weights")
        initial = np.zeros(len(weights), dtype=np.int8) if initial_mask is None else \
            validate_mask(initial_mask, weights, u, v)
        if not len(weights):
            return initial, dict(online_ms=(time.perf_counter() - started) * 1000, raw_value=0.,
                                 accepted_value=0., seed=int(seed), **self.setup)
        import scipy.sparse
        torch = self.torch
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        # The official MISDataset concatenates edges, reverse edges, and self
        # loops in this order. Self loops belong to model input, not feasibility.
        edges = np.column_stack((u, v))
        loops = np.column_stack((np.arange(len(weights)), np.arange(len(weights))))
        indices = np.concatenate((edges, edges[:, ::-1], loops), axis=0).T
        edge_index = torch.from_numpy(indices).long().to(self.device)
        adj_mat = scipy.sparse.coo_matrix((np.ones(indices.shape[1]), (indices[0], indices[1])),
                                         shape=(len(weights), len(weights)))
        self._sync()
        prep_ended = time.perf_counter()
        solutions = []
        with torch.no_grad():
            for _ in range(self.args.sequential_sampling):
                xt = torch.randn(len(weights), device=self.device)
                if self.args.parallel_sampling > 1:
                    xt = xt.repeat(self.args.parallel_sampling, 1, 1)
                    xt = torch.randn_like(xt)
                    current_edges = self.model.duplicate_edge_index(edge_index, len(weights), self.device)
                else:
                    current_edges = edge_index
                xt = xt.reshape(-1) if self.args.diffusion_type == "gaussian" else (xt > 0).long().reshape(-1)
                schedule = self.schedule_class(inference_schedule=self.args.inference_schedule,
                                               T=self.model.diffusion.T,
                                               inference_T=self.args.inference_diffusion_steps)
                for i in range(self.args.inference_diffusion_steps):
                    t1, t2 = schedule(i)
                    t1, t2 = np.asarray([t1], dtype=int), np.asarray([t2], dtype=int)
                    if self.args.diffusion_type == "categorical":
                        xt = self.model.categorical_denoise_step(xt, t1, self.device, current_edges,
                                                                target_t=t2)
                    else:
                        xt = self.model.gaussian_denoise_step(xt, t1, self.device, current_edges,
                                                             target_t=t2)
                # Preserve the official final heatmap transformation and CPU
                # decoder exactly. Unlike the upstream test hook, return masks.
                heatmap = xt.float().cpu().numpy()
                heatmap = heatmap + 1e-6 if self.args.diffusion_type == "categorical" else heatmap * .5 + .5
                for one_map in np.split(heatmap, self.args.parallel_sampling):
                    solutions.append(validate_mask(self.decode(one_map, adj_mat), weights, u, v))
        self._sync()
        infer_ended = time.perf_counter()
        best = max(solutions, key=lambda mask:float(weights @ mask))
        raw_value = float(weights @ best)
        retained = raw_value < float(weights @ initial) - 1e-9
        accepted = initial if retained else best
        details = dict(online_ms=(time.perf_counter() - started) * 1000,
                       preparation_ms=(prep_ended - started) * 1000,
                       sampling_and_decoding_ms=(infer_ended - prep_ended) * 1000,
                       raw_value=raw_value, accepted_value=float(weights @ accepted),
                       retained_incumbent=retained, native_warmstart=False, seed=int(seed), **self.setup)
        return accepted, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("chils", "difusco"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--binary")
    parser.add_argument("--checkout", default="tools/DIFUSCO-v3")
    parser.add_argument("--checkpoint", default="tools/DIFUSCO-v3/checkpoints/mis_sat_categorical.ckpt")
    parser.add_argument("--initial-mask")
    parser.add_argument("--seconds", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--weight-scale", type=int, default=1000)
    parser.add_argument("--alternating-seconds", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sequential-samples", type=int, default=1)
    parser.add_argument("--parallel-samples", type=int, default=1)
    parser.add_argument("--diffusion-type", choices=("categorical", "gaussian"), default="categorical")
    args = parser.parse_args()
    payload = np.load(args.input, allow_pickle=False)
    weights, u, v = validate_graph(payload["weights"], payload["edge_u"], payload["edge_v"])
    initial = np.load(args.initial_mask, allow_pickle=False)["selected"] if args.initial_mask else None
    if args.method == "chils":
        if not args.binary:
            parser.error("CHILS requires --binary")
        selected, details = solve_chils(weights, u, v, args.binary, args.seconds, args.seed, initial,
                                       args.population, args.threads, args.weight_scale, args.alternating_seconds)
    else:
        model = OfficialDIFUSCO(args.checkout, args.checkpoint, args.device, args.diffusion_type,
                               args.steps, args.sequential_samples, args.parallel_samples)
        selected, details = model.solve(weights, u, v, args.seed, initial)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(target) + ".npz", selected=selected)
    Path(str(target) + ".json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(dict(method=args.method, nodes=len(weights), edges=len(u),
                          value=float(weights @ selected), online_ms=details["online_ms"])))


if __name__ == "__main__":
    main()
