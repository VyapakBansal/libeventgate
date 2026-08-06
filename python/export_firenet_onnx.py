"""
ONLY Python required in Phase 0: load official FireNet .pth.tar → ONNX.
Everything else is C++/CUDA/TensorRT.

  cedric-scheerlinck/rpg_e2vid branch cedric/firenet
  checkpoint: firenet_1000.pth.tar
  Google Drive: https://drive.google.com/file/d/1nBCeIF_Us-rGhCjdU5q1Ch-yrFckjZPa

Exports voxel → frame with externalized ConvGRU states (h_in_*/h_out_*) so the
C++ IMU gate can freeze write-back without a custom TRT plugin.

  python python/export_firenet_onnx.py \\
    --repo third_party/rpg_e2vid \\
    --checkpoint weights/firenet_1000.pth.tar \\
    --out engines/firenet.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

StateT = Union[Any, Tuple[Any, ...]]


def _insert_repo(repo: Path) -> None:
    repo = repo.resolve()
    if not repo.is_dir():
        raise SystemExit(
            f"Missing FireNet repo at {repo}.\n"
            f"  bash scripts/fetch_firenet.sh"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def load_firenet(repo: Path, checkpoint: Path):
    """Load FireNet using official checkpoint layout (utils/loading_utils.load_model)."""
    _insert_repo(repo)
    import torch
    from model.model import E2VID, E2VIDRecurrent, FireNet  # noqa: F401

    raw = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    arch = raw.get("arch", "FireNet")
    try:
        model_type = raw["model"]
    except KeyError:
        model_type = raw["config"]["model"]

    factories = {
        "FireNet": FireNet,
        "E2VIDRecurrent": E2VIDRecurrent,
        "E2VID": E2VID,
    }
    if arch not in factories:
        raise SystemExit(f"Unsupported arch={arch!r}; expected one of {list(factories)}")

    print(f"arch={arch}")
    if isinstance(model_type, dict):
        print(f"model config keys: {sorted(model_type.keys())}")
        for k in (
            "num_bins",
            "base_num_channels",
            "num_residual_blocks",
            "recurrent_block_type",
            "skip_type",
            "num_encoders",
            "norm",
            "kernel_size",
            "recurrent_blocks",
        ):
            if k in model_type:
                print(f"  {k}: {model_type[k]}")

    model = factories[arch](model_type)
    state = raw.get("state_dict", raw)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing[:12]:", missing[:12])
    if unexpected:
        print("  unexpected[:12]:", unexpected[:12])
    model.eval()
    return model, model_type, arch


def _flatten_state_tensors(states: Sequence[StateT]) -> List[Any]:
    flat: List[Any] = []
    for s in states:
        if isinstance(s, tuple):
            flat.extend(list(s))
        else:
            flat.append(s)
    return flat


def _unflatten_states(flat: Sequence[Any], template: Sequence[StateT]) -> List[StateT]:
    out: List[StateT] = []
    i = 0
    for s in template:
        if isinstance(s, tuple):
            n = len(s)
            out.append(tuple(flat[i : i + n]))
            i += n
        else:
            out.append(flat[i])
            i += 1
    if i != len(flat):
        raise RuntimeError(f"state unflatten mismatch: used {i} of {len(flat)}")
    return out


def _discover_states(model, dummy_voxel, device) -> List[StateT]:
    """Dry-run with prev_states=None to learn rank/shape of each recurrent unit."""
    import torch

    with torch.no_grad():
        try:
            out = model(dummy_voxel, None)
        except TypeError:
            out = model(dummy_voxel)

    if not isinstance(out, (tuple, list)) or len(out) < 2:
        return []
    states = out[1]
    if states is None:
        return []
    if not isinstance(states, (list, tuple)):
        states = [states]

    result: List[StateT] = []
    for s in states:
        if isinstance(s, (list, tuple)):
            result.append(tuple(t.detach().clone().to(device) for t in s))
        else:
            result.append(s.detach().clone().to(device))
    return result


def _zero_like_states(states: Sequence[StateT], device) -> List[StateT]:
    import torch

    out: List[StateT] = []
    for s in states:
        if isinstance(s, tuple):
            out.append(tuple(torch.zeros_like(t, device=device) for t in s))
        else:
            out.append(torch.zeros_like(s, device=device))
    return out


def _make_external_wrapper(model, state_template: Sequence[StateT]):
    import torch.nn as nn

    class FireNetStepExternal(nn.Module):
        """
        Single step with explicit state I/O:
          in:  voxel, h_in_0 [, h_in_1 ...]
          out: frame, h_out_0 [, h_out_1 ...]

        ConvGRU → one tensor/unit. ConvLSTM → two tensors/unit (h, c) renamed sequentially.
        """

        def __init__(self, m, template: Sequence[StateT]):
            super().__init__()
            self.m = m
            self._template = template
            self.n_state_tensors = len(_flatten_state_tensors(template))

        def forward(self, voxel, *h_ins):
            if len(h_ins) != self.n_state_tensors:
                raise RuntimeError(
                    f"expected {self.n_state_tensors} state tensors, got {len(h_ins)}"
                )
            prev = _unflatten_states(list(h_ins), self._template)
            frame, new_states = self.m(voxel, prev)
            if new_states is None:
                return frame
            if not isinstance(new_states, (list, tuple)):
                new_states = [new_states]
            return (frame, *_flatten_state_tensors(new_states))

    return FireNetStepExternal(model, state_template)


def _make_internal_wrapper(model):
    """Fallback: voxel → frame only (zeros states each call — no true recurrence)."""
    import torch.nn as nn

    class FireNetStepInternal(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, voxel):
            try:
                out = self.m(voxel, None)
            except TypeError:
                out = self.m(voxel)
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

    return FireNetStepInternal(model)


def _export_onnx(wrapper, args_tuple, out_path: Path, input_names, output_names, opset: int) -> None:
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args_tuple,
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
        )


def _try_onnx_check(path: Path) -> None:
    try:
        import onnx

        m = onnx.load(str(path))
        onnx.checker.check_model(m)
        print(f"onnx checker: OK  ir_version={m.ir_version}")
        print("  inputs:")
        for i in m.graph.input:
            dims = [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim]
            print(f"    {i.name}: {dims}")
        print("  outputs:")
        for o in m.graph.output:
            dims = [d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim]
            print(f"    {o.name}: {dims}")
    except ImportError:
        print("onnx not installed — skip checker (optional: pip install onnx)")
    except Exception as e:  # noqa: BLE001
        print(f"onnx checker warning: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", type=Path, default=Path("third_party/rpg_e2vid"))
    ap.add_argument("--checkpoint", type=Path, default=Path("weights/firenet_1000.pth.tar"))
    ap.add_argument("--out", type=Path, default=Path("engines/firenet.onnx"))
    ap.add_argument("--height", type=int, default=512, help="IMX637 height")
    ap.add_argument("--width", type=int, default=640, help="IMX637 width")
    ap.add_argument("--bins", type=int, default=None, help="Override num_bins (default: checkpoint)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--externalize-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export h_in_*/h_out_* (default: on). Required for IMU gate freeze.",
    )
    ap.add_argument("--cpu", action="store_true", help="Force CPU export")
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(
            f"Missing checkpoint {args.checkpoint}\n"
            f"  Download firenet_1000.pth.tar from:\n"
            f"  https://drive.google.com/file/d/1nBCeIF_Us-rGhCjdU5q1Ch-yrFckjZPa\n"
            f"  Place in weights/firenet_1000.pth.tar"
        )

    import torch

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}")

    model, model_type, arch = load_firenet(args.repo, args.checkpoint)
    model.to(device)

    num_bins = int(model_type.get("num_bins", getattr(model, "num_bins", 5)))
    if args.bins is not None:
        num_bins = args.bins
        print(f"overriding bins={num_bins}")

    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"[vram] after load used={(total - free) / 1024**2:.0f}/{total / 1024**2:.0f} MiB")

    H, W = args.height, args.width
    dummy = torch.zeros(1, num_bins, H, W, device=device)

    state_template = _discover_states(model, dummy, device)
    n_units = len(state_template)
    n_tensors = len(_flatten_state_tensors(state_template)) if state_template else 0
    print(f"recurrent units={n_units}  flattened state tensors={n_tensors}")
    for i, s in enumerate(state_template):
        if isinstance(s, tuple):
            print(f"  unit[{i}] ConvLSTM-like shapes={[tuple(t.shape) for t in s]}")
        else:
            print(f"  unit[{i}] shape={tuple(s.shape)}")

    state_elems = 0
    for s in state_template:
        if isinstance(s, tuple):
            for t in s:
                state_elems += int(t.numel())
        else:
            state_elems += int(s.numel())
    print(
        f"[vram estimate] voxel≈{dummy.numel() * 4 / 1024**2:.1f} MiB  "
        f"states≈{state_elems * 4 / 1024**2:.1f} MiB  "
        f"(+ TRT workspace ≤512 MiB) — FireNet alone stays under ~4.5 GiB soft budget"
    )

    exported_state = False
    if args.externalize_state and n_tensors > 0:
        try:
            zeros = _zero_like_states(state_template, device)
            flat_in = _flatten_state_tensors(zeros)
            wrapper = _make_external_wrapper(model, state_template)
            wrapper.eval()
            input_names = ["voxel"] + [f"h_in_{i}" for i in range(n_tensors)]
            output_names = ["frame"] + [f"h_out_{i}" for i in range(n_tensors)]
            _export_onnx(
                wrapper, (dummy, *flat_in), args.out, input_names, output_names, args.opset
            )
            exported_state = True
            print(f"wrote {args.out}  (externalized {n_tensors} state tensors)")
            print("  gate: C++ restores h_in from hold on freeze; h_out → h_in when moving")
        except Exception as e:  # noqa: BLE001
            print(f"WARN: externalized export failed: {e}")
            print("  Falling back to voxel→frame only (gate freeze will be a no-op).")

    if not exported_state:
        wrapper = _make_internal_wrapper(model)
        wrapper.eval()
        _export_onnx(wrapper, (dummy,), args.out, ["voxel"], ["frame"], args.opset)
        print(f"wrote {args.out}  (frame-only; no state I/O)")
        print("NEXT: fix export so --externalize-state succeeds for gate freeze.")

    _try_onnx_check(args.out)
    print("\nNext:")
    print(
        "  ./build/build_engine --onnx engines/firenet.onnx "
        "--engine engines/firenet.engine --workspace-mb 512 --fp16 --no-int8"
    )
    print("  ./build/phase0_run --events ... --engine engines/firenet.engine --out out/recon")


if __name__ == "__main__":
    main()
