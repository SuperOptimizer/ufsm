#!/usr/bin/env python3
"""Wait for a surface-store build and qualify native training samples on the CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from build_surface_store import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/check_surface_samples"))
    parser.add_argument("--P", type=int, default=704)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--builder-unit", help="optional systemd user unit; detect failed builds while waiting")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = args.out / "qualification.json"
    try:
        while True:
            build = json.loads((args.work / "build.json").read_text())
            if build["status"] == "complete":
                break
            if not args.wait:
                raise RuntimeError("surface store is still building")
            if args.builder_unit:
                state = subprocess.check_output(["systemctl", "--user", "show", args.builder_unit,
                                                 "--property=ActiveState", "--value"], text=True).strip()
                if state not in {"active", "activating"}:
                    # A successful builder publishes build.json before exiting.
                    if json.loads((args.work / "build.json").read_text())["status"] == "complete":
                        continue
                    raise RuntimeError(f"builder stopped before publishing the store ({state})")
            time.sleep(30)
        config = json.loads(args.sources.read_text())
        if len(config["sources"]) != 1:
            raise ValueError("source configuration must refer to one merged volume")
        source = config["sources"][0]
        target = source["targets"]["recto"]
        store = Path(target["root"])
        if store.resolve() != Path(build["store"]).resolve() or target["group"] != ".":
            raise ValueError("source configuration does not refer to the completed store")
        provenance = json.loads((store / "provenance.json").read_text())
        if provenance["surface_segments"] != build["surface_segments"]:
            raise ValueError("store provenance does not match the inventoried surfaces")
        command = [str(args.binary.resolve()), str(args.sources.resolve()), str(args.out.resolve()),
                   str(args.P), str(args.batches)]
        atomic_json(report, {"status": "sampling", "surface_count": build["surface_count"], "command": command})
        with (args.out / "sampler.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        samples = json.loads((args.out / "samples.json").read_text())
        if not samples["passed"] or len(samples["samples"]) != args.batches:
            raise RuntimeError("native sample qualification failed")
        atomic_json(report, {"status": "complete", "surface_count": build["surface_count"],
                             "sources_sha256": hashlib.sha256(args.sources.read_bytes()).hexdigest(),
                             "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                             "command": command, **samples})
        print(f"SURFACE_SAMPLES_VERIFIED {report}", flush=True)
    except Exception as error:
        atomic_json(report, {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
