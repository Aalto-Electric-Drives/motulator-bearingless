"""
Solve the operating points of a plan in parallel.

Each worker holds one headless FEMM instance. The results are appended to a text file
one line per point, so that an interrupted run continues from where it stopped, and the
partial results can be used at any time.

Usage: python run_maps.py plan.npz results.txt [n_workers] [first_port]

"""

import multiprocessing as mp
import os
import sys
import time

import bsyrm_femm as bf
import numpy as np

RESTART_EVERY = 1000  # Solves between restarts of the FEMM instance
MAX_RESTARTS = 10  # Attempts to start a FEMM instance
PAR = bf.Params()
GEO = bf.build_geometry(PAR)  # Shared by the workers
_worker = {"count": 0}  # FEMM instance and its port in each worker process


def _init(queue) -> None:
    _worker["port"] = queue.get()
    _worker["model"] = bf.Model(PAR, _worker["port"], geo=GEO)


def _restart() -> None:
    """Restart the FEMM instance, waiting for the port to become free if needed."""
    try:
        _worker["model"].close()
    except Exception:  # noqa: BLE001
        pass
    for attempt in range(MAX_RESTARTS):
        try:
            _worker["model"] = bf.Model(PAR, _worker["port"], geo=GEO)
            return
        except OSError as ex:
            print(f"Port {_worker['port']}, restart {attempt}: {ex!r}", flush=True)
            time.sleep(5.0 * (attempt + 1))
    raise RuntimeError(f"FEMM does not start at the port {_worker['port']}")


def _task(args) -> str | None:
    """Solve one point and return the result line, or None if it fails twice."""
    k, i_s1, i_s2, q_M, theta_M = args
    _worker["count"] += 1
    if _worker["count"] % RESTART_EVERY == 0:
        _restart()
    for attempt in range(2):
        try:
            r = _worker["model"].solve(i_s1, i_s2, q_M, theta_M)
            vals = [
                r["psi_s1"].real,
                r["psi_s1"].imag,
                r["psi_s2"].real,
                r["psi_s2"].imag,
                r["f_M"].real,
                r["f_M"].imag,
                r["tau_M"],
                r["W"],
                r["W_co"],
                np.sum(r["psi_k"] * r["i_k"]),
                r["n_elem"],
            ]
            return f"{k} " + " ".join(f"{v:.12g}" for v in vals)
        except Exception as ex:  # noqa: BLE001
            print(
                f"Port {_worker['port']}, point {k}, attempt {attempt}: {ex!r}",
                flush=True,
            )
            _restart()
    return None


def main() -> None:
    """Run the plan."""
    plan = dict(np.load(sys.argv[1]))
    out = sys.argv[2]
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    first_port = int(sys.argv[4]) if len(sys.argv) > 4 else 49000
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {int(line.split()[0]) for line in f if line.strip()}
    n = plan["set"].size
    tasks = [
        (k, plan["i_s1"][k], plan["i_s2"][k], plan["q_M"][k], plan["theta_M"][k])
        for k in range(n)
        if k not in done
    ]
    print(f"{len(done)} points done, {len(tasks)} to go", flush=True)
    queue = mp.Queue()
    for port in range(first_port, first_port + n_workers):
        queue.put(port)
    t_0 = time.time()
    with mp.Pool(n_workers, _init, (queue,)) as pool, open(out, "a") as f:
        for count, line in enumerate(pool.imap(_task, tasks)):
            if line is not None:
                f.write(line + "\n")
            if count % 20 == 0:
                f.flush()
            if count % 200 == 0:
                rate = (count + 1) / (time.time() - t_0)
                eta = (len(tasks) - count) / rate / 3600
                print(
                    f"{count}/{len(tasks)}, {rate:.2f} /s, {eta:.2f} h left", flush=True
                )
    print("Finished", flush=True)


if __name__ == "__main__":
    main()
