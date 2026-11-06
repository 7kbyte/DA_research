#!/usr/bin/env python3
"""Read-only CPU export heartbeat; never opens model evaluation scores."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/"results/primary_remaining_endpoints_export_state.json"
OUTPUT=ROOT/"results/primary_remaining_endpoints_export_monitor.json"


def write(value):
    temporary=OUTPUT.with_name(OUTPUT.name+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(OUTPUT)


def main():
    seen=set();events=[]
    while True:
        state=json.loads(STATE.read_text())
        alive=Path(f"/proc/{state['pid']}/cmdline").exists()
        if alive:
            command=Path(f"/proc/{state['pid']}/cmdline").read_bytes()
            alive=b"export_remaining_primary_endpoints.py" in command
        now=dt.datetime.now(dt.timezone.utc)
        record=dict(updated_at_utc=now.isoformat(),monitor_pid=os.getpid(),export_pid=state["pid"],export_process_alive=alive,
            export_status=state["status"],stage=state["stage"],current_endpoint_id=state.get("current_endpoint_id"),
            completed_unique_weights=state["completed_unique_weights"],expected_unique_weights=45,
            seconds_since_last_export_stage=(now-dt.datetime.fromisoformat(state["updated_at_utc"])).total_seconds(),
            source_state_sha256=hashlib.sha256(STATE.read_bytes()).hexdigest(),
            completed_packages=state.get("completed_packages",[]),gpu_used=False,shared_storage_written=False)
        for package in state.get("completed_packages",[]):
            if package["tar_path"] not in seen:
                event=dict(event="new_verified_scratch_package",detected_at_utc=now.isoformat(),tar_path=package["tar_path"],
                    tar_bytes=package["tar_bytes"],tar_sha256=package["tar_sha256"],logical_endpoints=package["logical_endpoints"],
                    verified_unique_weights=package["verified_unique_weights"],alias_count=package["alias_count"])
                events.append(event);seen.add(package["tar_path"])
                print(json.dumps(event),flush=True)
        record["events"]=events
        failure=state["status"]=="failed" or (not alive and state["status"]!="passed")
        record["status"]="failed_requires_review" if failure else "passed" if state["status"]=="passed" else "monitoring"
        if failure:
            record["error"]=state.get("error","export process terminated without completion receipt")
            print(json.dumps(dict(event="failed_requires_review",error=record["error"])),flush=True)
        write(record)
        if failure or state["status"]=="passed":return
        time.sleep(30)


if __name__=="__main__":main()
