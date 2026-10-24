# NUMA placement at process startup

`resin.numa` places a worker's CPU execution and new host-memory allocations near
its GPU. Bind **before starting the Python child**: imports, allocators and
communication libraries may allocate memory or create threads before the worker
function runs. Binding inside that function misses this startup work.

## Start one worker per GPU

`resin.process.spawn` starts fresh Python interpreters and applies placement
before each one starts. The worker receives its rank followed by `args`:

```python
import resin


def worker(rank, config):
    import torch

    device = config["devices"][rank]
    torch.cuda.set_device(device)
    # Initialize process groups and construct the model runner here.


if __name__ == "__main__":
    devices = ["cuda:0", "cuda:1"]
    workers = resin.process.spawn(
        worker,
        args=({"devices": devices},),
        devices=devices,
        numa="auto",
    )
```

Keep the worker at module scope and guard process startup with
`if __name__ == "__main__"`. `args` and the target must be picklable, as with
Python's multiprocessing `spawn` start method. Choose the worker's GPU from the
same rank-to-device mapping passed to `resin.process.spawn`.

| `numa` | Behavior |
| --- | --- |
| `"auto"` | Bind where supported; warn and fall back to CPU affinity only or no placement when necessary. |
| `"off"` | Start workers without changing placement. |
| `"required"` | Require the requested placement; fail if it cannot be applied. |

`join=True` is the default: startup blocks until the workers finish, then returns
a `resin.process.ProcessContext`. With `join=False`, use `context.join(timeout)`
to wait; it returns whether all workers have finished. `context.processes`
exposes the child processes. Call `close()` to clean up remaining workers, or
use the context as a context manager.

If a worker fails, joining stops its peers and raises
`resin.process.ProcessError`.

## Use Python multiprocessing directly

The lower-level context applies to the calling thread. A directly spawned child
inherits its CPU affinity and memory policy before Python begins executing:

```python
import multiprocessing

import resin


def worker(device):
    import torch

    torch.cuda.set_device(device)
    # Load weights and initialize the runner here.


if __name__ == "__main__":
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=worker, args=("cuda:0",))
    with resin.numa.bind(device="cuda:0") as placement:
        print(placement)
        process.start()
    process.join()
```

Leaving `bind` restores the calling thread's previous affinity and memory policy;
the child retains what it inherited. Start each rank inside its own binding
context. This mechanism does not place workers launched by an already running
`forkserver`, a remote launcher, or another process.

You can also select a PCI device or provide the NUMA node and CPUs explicitly:

```python
with resin.numa.bind(device="pci:0000:ab:00.0") as placement:
    process.start()

# Choose CPUs that are available to this process on the requested node.
with resin.numa.bind(node=1, cpus=range(32, 40), strict=True) as placement:
    process.start()
```

These are alternative startup examples; a multiprocessing process can start only
once. GPU lookup honors CUDA device visibility without importing Torch or
creating a CUDA context. Placement uses Linux interfaces directly and does not
require `numactl` or `libnuma`.

## Observe placement and its limits

The yielded binding records `device`, `node`, `cpus`, `memory_policy` and `reason`.
Inspect these fields when recording benchmark provenance. The default memory
policy prefers the selected node, allowing allocations to fall back to another
node when necessary. `strict=False`, the default, warns when discovery or policy
application is unavailable and records the fallback reason. `strict=True` raises
instead, including on unsupported platforms or when the OS denies the policy.

Affinity constrains where threads can run; it does not reserve CPU cores or give
workers exclusive access. Workers near the same GPU socket may share CPUs. Allow
room for communication helpers, HTTP serving and tokenization rather than
restricting an entire multithreaded worker to one core.

Placement influences new allocations. It does not move existing pages, relocate
shared memory created by another process, or place GPU HBM. Child-created threads
inherit the child's initial settings; existing parent threads are unaffected.
Container CPU masks can restrict eligible CPUs, and a CPU quota can limit actual
CPU capacity independently of that mask. GPU-to-NUMA topology may be hidden or
unavailable in a container; inspect the binding result rather than assuming that
a successful worker launch implies full placement.
