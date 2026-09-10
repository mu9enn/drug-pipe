"""Resume large official runtime wheels independently of an installer retry."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aria2", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    response = requests.get("https://pypi.org/pypi/torch/2.9.1/json", timeout=60)
    response.raise_for_status()
    packages = {"nvidia-nccl-cu12":"2.27.5", "torch":"2.9.1", "vllm":"0.14.1"}
    for requirement in response.json()["info"]["requires_dist"]:
        if requirement.startswith("nvidia-"):
            name, version = requirement.split(";", 1)[0].strip().split("==")
            packages[name] = version

    def download(item):
        name, version = item
        response = requests.get(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60)
        response.raise_for_status()
        wheels = [w for w in response.json()["urls"] if "x86_64" in w["filename"]
                  and any(tag in w["filename"] for tag in ("-cp311-cp311-", "-cp38-abi3-", "-py3-none-"))]
        assert len(wheels) == 1, (name,[w["filename"] for w in wheels])
        wheel = wheels[0]
        destination = root / wheel["filename"]
        env = os.environ.copy()
        for key in ("HTTP_PROXY", "HTTPS_PROXY"):
            if key in env:
                env[key.lower()] = env[key]
        library_dir = args.aria2.resolve().parent.parent / "lib/x86_64-linux-gnu"
        env["LD_LIBRARY_PATH"] = str(library_dir) + ":" + env.get("LD_LIBRARY_PATH", "")
        with (root / f"{name}.download.log").open("a") as log:
            subprocess.run([str(args.aria2.resolve()), "--continue=true", "--max-tries=20", "--timeout=120",
                            "--max-connection-per-server=8", "--split=8", "--min-split-size=4M",
                            "--file-allocation=none", "--auto-file-renaming=false", "--summary-interval=60",
                            "--console-log-level=warn", "--check-integrity=true",
                            "--checksum=sha-256="+wheel["digests"]["sha256"],
                            "--dir="+str(root), "--out="+wheel["filename"], wheel["url"]],
                           env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        digest = hashlib.sha256()
        with destination.open("rb") as source:
            for block in iter(lambda:source.read(8*1024**2),b""):
                digest.update(block)
        assert digest.hexdigest() == wheel["digests"]["sha256"], name
        print("verified",name,version,flush=True)
        return {"name":name,"version":version,"path":str(destination),"url":wheel["url"],"sha256":digest.hexdigest()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download,packages.items()))
    (root / "verified_manifest.json").write_text(json.dumps(records,indent=2)+"\n")


if __name__ == "__main__":
    main()
