#!/usr/bin/env python3
"""Run an auditable, reloadable single-GPU job queue within a hard deadline.

Queue JSON is a list of {id, argv, depends_on?, env?, expected_seconds?, outputs?}.
No shell interpolation is performed. A queue may be atomically extended while
running; existing job definitions cannot change. Work runs only on listed GPUs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_time(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def validate_queue(jobs):
    if not isinstance(jobs, list):
        raise ValueError("queue must be a list")
    ids = set()
    for job in jobs:
        name = job.get("id")
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-." for c in name):
            raise ValueError(f"invalid job id: {name!r}")
        if name in ids:
            raise ValueError(f"duplicate job id: {name}")
        ids.add(name)
        if not isinstance(job.get("argv"), list) or not job["argv"] or not all(isinstance(x, str) for x in job["argv"]):
            raise ValueError(f"job {name} requires a nonempty list of string argv")
        if float(job.get("expected_seconds", 0)) < 0:
            raise ValueError("expected_seconds must be nonnegative")
    for job in jobs:
        if any(x not in ids for x in job.get("depends_on", [])):
            raise ValueError(f"unknown dependency in {job['id']}")
    visiting, done = set(), set()
    lookup = {x["id"]: x for x in jobs}
    def visit(name):
        if name in visiting:
            raise ValueError("dependency cycle")
        if name in done:
            return
        visiting.add(name)
        for dependency in lookup[name].get("depends_on", []):
            visit(dependency)
        visiting.remove(name)
        done.add(name)
    for name in lookup:
        visit(name)
    return jobs


def digest_job(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--state", type=Path, default=Path("logs/job_state.json"))
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--deadline", required=True)
    p.add_argument("--launch-margin-seconds", type=float, default=1800)
    p.add_argument("--stop-margin-seconds", type=float, default=600)
    p.add_argument("--terminate-grace-seconds", type=float, default=90)
    p.add_argument("--poll-seconds", type=float, default=10)
    p.add_argument("--exit-when-idle", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.state.parent / "jobs"
    log_dir.mkdir(exist_ok=True)
    lock_handle = args.state.with_suffix(".lock").open("w")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_handle.write(str(os.getpid()))
    lock_handle.flush()
    deadline = parse_time(args.deadline)
    gpu_ids = args.gpus.split(",")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("duplicate GPUs")
    state = json.loads(args.state.read_text()) if args.state.exists() else {"jobs": {}}
    for name, record in state["jobs"].items():
        if record["status"] == "running":
            # Never adopt an arbitrary PID or duplicate a potentially live job.
            raise RuntimeError(f"state has running job {name}; reconcile its PID before restarting")
    active = {}
    stopped = []
    def stop(signum, frame):
        stopped.append(signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    event_path = args.state.parent / "scheduler_events.jsonl"
    def event(kind, **fields):
        item = {"utc": utc_now(), "event": kind, **fields}
        with event_path.open("a") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(json.dumps(item, ensure_ascii=False), flush=True)
    event("scheduler_start", pid=os.getpid(), deadline=args.deadline, gpus=gpu_ids)
    while True:
        now = time.time()
        queue = validate_queue(json.loads(args.queue.read_text()))
        for job in queue:
            name, digest = job["id"], digest_job(job)
            if name in state["jobs"] and state["jobs"][name]["definition_sha256"] != digest:
                raise RuntimeError(f"job definition changed: {name}; use a new id")
            state["jobs"].setdefault(name, {"status": "pending", "definition_sha256": digest})
        stopping = bool(stopped) or now >= deadline - args.stop_margin_seconds
        for name, (process, handle, job) in list(active.items()):
            record = state["jobs"][name]
            code = process.poll()
            if code is None and stopping and "stop_requested_at" not in record:
                os.killpg(process.pid, signal.SIGTERM)
                record["stop_requested_at"] = now
                event("job_stop_requested", id=name, pid=process.pid)
            elif code is None and "stop_requested_at" in record and now - record["stop_requested_at"] >= args.terminate_grace_seconds:
                os.killpg(process.pid, signal.SIGKILL)
            if code is not None:
                handle.close()
                missing = [x for x in job.get("outputs", []) if not Path(x).exists()]
                record.update(status="completed" if code == 0 and not missing and "stop_requested_at" not in record else "failed",
                              returncode=code, ended_at_utc=utc_now(), elapsed_seconds=now-record["started_at_epoch"],
                              missing_outputs=missing)
                event("job_finish", id=name, **record)
                del active[name]
        busy = {state["jobs"][name]["gpu"] for name in active}
        for job in queue:
            name = job["id"]
            record = state["jobs"][name]
            if record["status"] != "pending":
                continue
            dependencies = [state["jobs"][x]["status"] for x in job.get("depends_on", [])]
            if any(x in {"failed", "blocked"} for x in dependencies):
                record.update(status="blocked", reason="dependency_failed")
                event("job_blocked", id=name)
                continue
            if stopping or any(x != "completed" for x in dependencies):
                continue
            if now + float(job.get("expected_seconds", 0)) >= deadline - args.launch_margin_seconds:
                continue
            available = [x for x in gpu_ids if x not in busy]
            if not available:
                break
            gpu = available[0]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1",
                   "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false",
                   "HF_HOME": "/tmp/grad-research-20261106/cache/huggingface",
                   "HF_ENDPOINT": "https://hf-mirror.com", "HF_HUB_DISABLE_TELEMETRY": "1",
                   **job.get("env", {})}
            log_path = log_dir / f"{name}.log"
            handle = log_path.open("ab", buffering=0)
            process = subprocess.Popen(job["argv"], cwd=root, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record.update(status="running", pid=process.pid, gpu=gpu, log=str(log_path),
                          started_at_utc=utc_now(), started_at_epoch=time.time(), argv=job["argv"])
            active[name] = (process, handle, job)
            busy.add(gpu)
            event("job_start", id=name, gpu=gpu, pid=process.pid, argv=job["argv"])
        state.update(updated_at_utc=utc_now(), scheduler_pid=os.getpid(), deadline_utc=args.deadline,
                     active_jobs=list(active), stopping=stopping)
        atomic_json(args.state, state)
        if stopping and not active:
            break
        if args.exit_when_idle and not active and all(x["status"] != "pending" for x in state["jobs"].values()):
            break
        time.sleep(min(args.poll_seconds, 30))
    event("scheduler_stop", active_jobs=list(active))


if __name__ == "__main__":
    main()
