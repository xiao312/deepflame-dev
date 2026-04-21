#!/usr/bin/env python3
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import cantera as ct
    import torch
except Exception as exc:
    raise SystemExit(
        "ERROR: build_h2_onnx_wrapper.py requires cantera and torch in the active Python environment.\n"
        f"Original import error: {exc}"
    )

try:
    import onnx  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "ERROR: build_h2_onnx_wrapper.py requires the 'onnx' Python package for torch.onnx.export.\n"
        f"Original import error: {exc}"
    )


@dataclass
class TorchSettings:
    model_name: str
    mechanism_name: str
    frozen_temperature: float
    delta_t: float


def parse_cantera_torch_properties(case_dir: Path) -> TorchSettings:
    text = (case_dir / "constant" / "CanteraTorchProperties").read_text()

    def read_quoted(key: str) -> str:
        m = re.search(rf"{key}\s+\"([^\"]+)\"", text)
        if not m:
            raise RuntimeError(f"missing quoted key: {key}")
        return m.group(1)

    def read_scalar(key: str) -> float:
        m = re.search(rf"{key}\s+([^;]+);", text)
        if not m:
            raise RuntimeError(f"missing scalar key: {key}")
        return float(m.group(1).strip())

    return TorchSettings(
        model_name=read_quoted("torchModel"),
        mechanism_name=read_quoted("CanteraMechanismFile"),
        frozen_temperature=read_scalar("frozenTemperature"),
        delta_t=read_scalar("inferenceDeltaTime"),
    )


class NN_MLP(torch.nn.Module):
    def __init__(self, layer_info):
        super().__init__()
        self.net = torch.nn.Sequential()
        n = len(layer_info) - 1
        for i in range(n - 1):
            self.net.add_module(f"linear_layer_{i}", torch.nn.Linear(layer_info[i], layer_info[i + 1]))
            self.net.add_module(f"gelu_layer_{i}", torch.nn.GELU())
        self.net.add_module(f"linear_layer_{n - 1}", torch.nn.Linear(layer_info[n - 1], layer_info[n]))

    def forward(self, x):
        return self.net(x)


class H2InferenceWrapper(torch.nn.Module):
    def __init__(self, case_dir: Path):
        super().__init__()
        cfg = parse_cantera_torch_properties(case_dir)
        mech_path = case_dir / cfg.mechanism_name
        model_path = case_dir / cfg.model_name
        if not mech_path.exists():
            raise RuntimeError(f"mechanism file not found: {mech_path}")
        if not model_path.exists():
            raise RuntimeError(f"model file not found: {model_path}")

        gas = ct.Solution(str(mech_path))
        n_species = gas.n_species
        state_dict = torch.load(model_path, map_location="cpu")

        self.n_species = n_species
        self.feature_dim = 3 + n_species

        self.register_buffer("xmu", torch.tensor(state_dict["data_in_mean"], dtype=torch.float64).unsqueeze(0))
        self.register_buffer("xstd", torch.tensor(state_dict["data_in_std"], dtype=torch.float64).unsqueeze(0))
        self.register_buffer("ymu", torch.tensor(state_dict["data_target_mean"], dtype=torch.float64).unsqueeze(0))
        self.register_buffer("ystd", torch.tensor(state_dict["data_target_std"], dtype=torch.float64).unsqueeze(0))
        self.register_buffer("frozen_temperature", torch.tensor(cfg.frozen_temperature, dtype=torch.float64))
        self.register_buffer("pressure_scale", torch.tensor(101325.0, dtype=torch.float64))
        self.register_buffer("delta_t", torch.tensor(cfg.delta_t, dtype=torch.float64))
        self.register_buffer("lambda_bct", torch.tensor(0.1, dtype=torch.float64))

        layers = [n_species + 2, 1600, 800, 400, 1]
        self.models = torch.nn.ModuleList()
        for i in range(n_species - 1):
            model = NN_MLP(layers)
            model.load_state_dict(state_dict[f"net{i}"])
            model.eval()
            self.models.append(model)

    def forward(self, raw_input):
        x = torch.abs(raw_input).to(torch.float64)
        x = x.clone()
        x[:, 1] = x[:, 1] * self.pressure_scale

        mask = (x[:, 0] > self.frozen_temperature).unsqueeze(1).to(torch.float64)
        rho0 = x[:, -1:].clone()
        input_y = x[:, 2:-1].clone()
        input_bct = x[:, 0:-1].clone()
        input_bct[:, 2:] = (torch.pow(input_bct[:, 2:], self.lambda_bct) - 1.0) / self.lambda_bct
        input_normalized = ((input_bct - self.xmu) / self.xstd).to(torch.float32)

        outputs = [model(input_normalized) for model in self.models]
        output_normalized = torch.cat(outputs, dim=1)

        output_bct = output_normalized.to(torch.float64) * self.ystd + self.ymu + input_bct[:, 2:-1]
        output_y = input_y.clone()
        output_y[:, :-1] = torch.pow(self.lambda_bct * output_bct + 1.0, 1.0 / self.lambda_bct)
        output_y[:, :-1] = (
            output_y[:, :-1]
            / torch.sum(output_y[:, :-1], dim=1, keepdim=True)
            * (1.0 - output_y[:, -1:])
        )
        output = (output_y - input_y) * rho0 / self.delta_t
        return (output * mask).to(torch.float32)


def main():
    case_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    out_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else (case_dir / "h2_full_inference_wrapper.onnx")

    wrapper = H2InferenceWrapper(case_dir)
    wrapper.eval()

    # Representative dummy batch for export shape inference. Runtime batch size
    # stays dynamic through dynamic_axes below.
    dummy = torch.ones((8, wrapper.feature_dim), dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["cell_inputs"],
        output_names=["cell_outputs"],
        dynamic_axes={"cell_inputs": {0: "num_cells"}, "cell_outputs": {0: "num_cells"}},
        opset_version=17,
    )
    print(f"Wrote ONNX wrapper to {out_path}")


if __name__ == "__main__":
    main()
