import gc
import os
import sys

# TF32 mode, same switch as constraintflow-gpu's cli.py: CF_DISABLE_TF32=1 forces full fp32
# (PyTorch lets cuDNN convolutions use TF32 by default). NVIDIA_TF32_OVERRIDE must be set
# before the CUDA libraries load.
DISABLE_TF32 = os.environ.get("CF_DISABLE_TF32", "0") not in ("", "0")
if DISABLE_TF32:
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import torch
if DISABLE_TF32:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

#FEAT: --dtype float64 (fp64 precision oracle) must be applied before modules that read torch.float at import.
from constraintflow.lib.precision import dtype_from_argv, use_float64
DTYPE = dtype_from_argv(sys.argv[1:])
if DTYPE not in ("float32", "float64"):
    sys.exit(f"Error: --dtype must be float32 or float64, got {DTYPE!r}")
if DTYPE == "float64":
    use_float64()
import typer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from constraintflow.compiler.compile import compile as _compile
from constraintflow.verifier.provesound import provesound as _provesound
from constraintflow.lib.globals import set_device

app = typer.Typer(help="ConstraintFlow CLI for verification and compilation of DSL programs.")


# --------------------------
# Utility Functions
# --------------------------

def get_program(program_file: str) -> str:
    return program_file

def get_network(network: str, network_format: str, dataset: str) -> str:
    if dataset not in ["mnist", "cifar"]:
        return network
    return network



def get_dataset(batch_size: int, dataset: str, train: bool = False):
    if dataset == "mnist":
        transform = transforms.ToTensor()  # keep 28x28
        data = datasets.MNIST(root=".", train=train, download=True, transform=transform)
    elif dataset == "cifar10" or dataset == "cifar":
        transform = transforms.ToTensor()  # keep 32x32
        data = datasets.CIFAR10(root=".", train=train, download=True, transform=transform)
    elif dataset == "tinyimagenet":
        train = True
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # TinyImageNet images are 64x64
            transforms.ToTensor(),
        ])
        root_dir = "tinyimagenet/tiny-imagenet-200"
        split = "train" if train else "test"
        data_dir = os.path.join(root_dir, split)
        if train:
            data = datasets.ImageFolder(root=data_dir, transform=transform)
        else:
            # TinyImageNet test: all images in one folder
            from torchvision.datasets.folder import default_loader
            class TinyImageNetTest(torch.utils.data.Dataset):
                def __init__(self, root, transform=None):
                    self.root = root
                    self.transform = transform
                    self.loader = default_loader
                    self.images = sorted(os.listdir(root))
                def __len__(self):
                    return len(self.images)
                def __getitem__(self, idx):
                    path = os.path.join(self.root, self.images[idx])
                    img = self.loader(path)
                    if self.transform:
                        img = self.transform(img)
                    return img, -1
            data = TinyImageNetTest(data_dir, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    image, label = next(iter(dataloader))
    if dataset == 'tinyimagenet':
        image = image[:, :, :56, :56]  # ensure 3 channels
    # ensure labels are a tensor
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
    return image, label



def get_precision(lb):
    verified = (lb >= 0).all(dim=1)
    precision = verified.sum() / verified.shape[0]
    return precision


# --------------------------
# CLI Commands
# --------------------------

@app.command()
def provesound(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    nprev: int = typer.Option(1, help="Number of previous states"),
    nsymb: int = typer.Option(1, help="Number of symbols"),
):
    """
    Prove soundness of a ConstraintFlow program.
    """
    program = get_program(program_file)
    res = _provesound(program, nprev=nprev, nsymb=nsymb)
    typer.echo(f"Provesound result: {res}")


def compile_code(
    program_file: str,
    output_path: str = "output/",
    no_rewrite: bool = False,
):
    """
    Compile a ConstraintFlow program into Python.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
    except OSError as e:
        typer.echo(f"Error creating folder '{output_path}': {e}")
        raise typer.Exit(code=1)

    program = get_program(program_file)
    res = _compile(program, output_path, no_rewrite=no_rewrite)
    if res:
        typer.echo("Compilation successful ✅")
    else:
        typer.echo("Compilation failed ❌")
        raise typer.Exit(code=1)

@app.command()
def compile(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    output_path: str = typer.Option("output/", help="Output path for generated code"),
    no_rewrite: bool = typer.Option(False, help="Skip domain-specific rewrite pass (ablation N-R)"),
    #FIX FOR: no way to choose center/radius vs the old sign-split concretization at compile time.
    center_radius: bool = typer.Option(True, "--center-radius/--no-center-radius", help="Concretize against n[l]/n[u] as c*mid + |c|*rad (constraintflow-gpu's arithmetic) instead of the sign split c+*u + c-*l, which loses interval width in fp32 on narrow boxes; costs extra eager ops."),
):
    from constraintflow.lib.globals import center_radius as center_radius_flag
    center_radius_flag.set_flag() if center_radius else center_radius_flag.reset_flag()
    compile_code(program_file, output_path, no_rewrite=no_rewrite)


@app.command()
def run(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    network: str = typer.Option("mnist_relu_3_50", help="Network name"),
    network_format: str = typer.Option("onnx", help="Network format"),
    dataset: str = typer.Option("mnist", help="Dataset (mnist or cifar)"),
    batch_size: int = typer.Option(1, help="Batch size"),
    eps: float = typer.Option(0.01, help="Epsilon"),
    train: bool = typer.Option(False, help="Run on training dataset"),
    print_intermediate_results: bool = typer.Option(False, help="Print intermediate results"),
    no_sparsity: bool = typer.Option(False, help="Disable sparsity optimizations"),
    output_path: str = typer.Option("output/", help="Path where compiled program is stored"),
    compile: bool = typer.Option(False, help="Run compilation before execution"),
    device: str = typer.Option("cpu", help="Device to run on (cpu or cuda)"),
    warmup: int = typer.Option(0, min=0, help="Untimed warmup runs on different data before the timed runs"),
    repeat: int = typer.Option(1, min=1, help="Number of timed runs"),
    num_images: int = typer.Option(0, "--num-images", min=0, help="Verify this many test images, batch-size at a time, one timed run per batch (batch i = images [i*batch_size, (i+1)*batch_size)); warmup j uses the images right after them. Requires --results-file; replaces --repeat. 0 (default): the --repeat protocol, every timed run on the first batch."),
    results_file: str = typer.Option(None, "--results-file", help="With --num-images: write one JSON line per warmup and per timed batch (time, peak/resident memory, bounds) here, outside the timer."),
    #FEAT: --dtype on run, so typer accepts it (applied from argv above).
    dtype: str = typer.Option("float32", "--dtype", help="float32, or float64 to compute every float tensor in fp64 (a precision oracle for bound mismatches). Read from argv before the imports; listed here so typer accepts it."),
    #FEAT: --profile-comm records CPU-GPU communication per timed batch (lib/comm_profile.py).
    profile_comm: bool = typer.Option(False, "--profile-comm", help="With --num-images: profile each timed batch with torch.profiler and add syncs, alloc_syncs, gpu_idle_pct and sync_calls to its results line; its time goes to profiled_time_s instead of time_s. Warmups are not profiled."),
):
    """
    Run a compiled ConstraintFlow program.
    """
    if num_images and (num_images % batch_size or not results_file):
        typer.echo(f"Error: --num-images ({num_images}) must be a multiple of --batch-size ({batch_size}) and needs --results-file.")
        raise typer.Exit(code=1)
    if profile_comm and not num_images:
        typer.echo("Error: --profile-comm requires --num-images.")
        raise typer.Exit(code=1)
    set_device(device)

    try:
        os.makedirs(output_path, exist_ok=True)
    except OSError as e:
        typer.echo(f"Error creating folder '{output_path}': {e}")
        raise typer.Exit(code=1)

    if compile:
        compile_code(program_file, output_path)

    sys.path.insert(0, os.path.abspath(output_path))
    from main import run  # compiled code provides this

    network_file = get_network(network, network_format, dataset)
    # Images stay on the host; each run moves only its own batch.
    X, y = get_dataset(num_images + warmup * batch_size if num_images else batch_size * (warmup + 1),
                       dataset, train=train)

    import time
    is_cuda = torch.device(device).type == "cuda"
    mem_label = "Peak GPU memory" if is_cuda else "Peak CPU memory"

    def _call(lo, hi, print_intermediate):
        return run(network_file, batch_size, eps, X[lo:hi].to(device), y[lo:hi].to(device), dataset=dataset,
                   train=train, print_intermediate_results=print_intermediate, no_sparsity=no_sparsity)

    if num_images:
        _run_image_batches(_call, batch_size, num_images, warmup, results_file, is_cuda,
                           print_intermediate_results, profile_comm)
        return

    for i in range(warmup):
        t0 = time.perf_counter()
        _call((i + 1) * batch_size, (i + 2) * batch_size, False)
        if is_cuda:
            torch.cuda.synchronize()
        typer.echo(f"Warmup run {i + 1}/{warmup}: {time.perf_counter() - t0:.6f} s")

    # Freeze the startup heap so automatic collections during timed runs only scan per-run garbage.
    gc.collect()
    gc.freeze()

    times, peaks = [], []
    for i in range(repeat):
        resident = 0
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        lb, ub = _call(0, batch_size, print_intermediate_results)
        if is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if is_cuda:
            peak = torch.cuda.max_memory_allocated()
        else:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        peaks.append(peak)
        resident_note = f", Resident GPU memory: {resident} bytes" if is_cuda else ""
        typer.echo(f"Run {i + 1}/{repeat}: {times[-1]:.6f} s, {mem_label}: {peak} bytes{resident_note}")

    # Full float32 precision (9 significant digits) and no '...' elision, so the harness
    # can parse the exact bounds into its CSV.
    torch.set_printoptions(precision=8, sci_mode=True, threshold=sys.maxsize)
    typer.echo(f"Lower bound: {lb}")
    typer.echo(f"Upper bound: {ub}")
    torch.set_printoptions(profile="default")
    precision = get_precision(lb)
    typer.echo(f"Precision: {precision}")
    typer.echo(f"Average verification time: {sum(times) / repeat:.6f}")
    typer.echo(f"{mem_label}: {max(peaks)} bytes")


def _run_image_batches(_call, batch_size, num_images, warmup, results_file, is_cuda,
                       print_intermediate_results, profile_comm=False):
    """The --num-images protocol: warmups on the images after the evaluated ones, then one
    timed run per batch. The timed region is the --repeat protocol's (host-to-device copy,
    bounds, sync); the results line is written after the timer stops. No error handling:
    a failing batch kills the process, and the batches already written stay in the file."""
    import json
    import time
    #FEAT: --profile-comm profiles each timed region.
    from constraintflow.lib.comm_profile import CommProfile

    def write(f, record):
        f.write(json.dumps(record) + "\n")
        f.flush()

    with open(results_file, "w") as out:
        for j in range(warmup):
            lo = num_images + j * batch_size
            t0 = time.perf_counter()
            _call(lo, lo + batch_size, False)
            if is_cuda:
                torch.cuda.synchronize()
            warmup_time = time.perf_counter() - t0
            typer.echo(f"Warmup run {j + 1}/{warmup}: {warmup_time:.6f} s")
            write(out, {"kind": "warmup", "index": j, "images": [lo, lo + batch_size],
                        "time_s": warmup_time})

        # Freeze the startup heap so automatic collections during timed runs only scan per-run garbage.
        gc.collect()
        gc.freeze()

        n_batches = num_images // batch_size
        for i in range(n_batches):
            lo = i * batch_size
            typer.echo(f"=== batch {i} start ===")
            resident = None
            #FEAT: profiler on around the timer; comm.region() is exactly the timed region.
            comm = CommProfile(profile_comm)
            with comm:
                if is_cuda:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    resident = torch.cuda.memory_allocated()
                t0 = time.perf_counter()
                with comm.region():
                    lb, ub = _call(lo, lo + batch_size, print_intermediate_results)
                    if is_cuda:
                        torch.cuda.synchronize()
                run_time = time.perf_counter() - t0
            # Per-run peaks exist only on the GPU; a CPU process's peak RSS is lifetime-wide.
            peak = torch.cuda.max_memory_allocated() if is_cuda else None
            write(out, {"kind": "batch", "batch": i, "images": [lo, lo + batch_size],
                        comm.time_key: run_time, "peak_bytes": peak, "resident_bytes": resident,
                        "lower": lb.detach().cpu().tolist(), "upper": ub.detach().cpu().tolist(),
                        **comm.metrics()})
            comm.save_trace(results_file, i)
            typer.echo(f"Run {i + 1}/{n_batches}: {run_time:.6f} s, Peak memory: {peak} bytes, "
                       f"Resident memory: {resident} bytes")
            typer.echo(f"=== batch {i} end ===")


def main():
    if '-O' in sys.argv:
        sys.argv.remove('-O')
        os.execv(sys.executable, [sys.executable, '-O'] + sys.argv)
    app()


if __name__ == "__main__":
    main()
