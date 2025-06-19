from scaled_curve_helpers import scale_to_gpt4o, curve_for_model, plot_curves
from inference_economics_notebook import DeepSeek_V3, Llama_3_405B, GPT_4
import argparse

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(x, *args, **kwargs):
        return x

models = {
    "DeepSeek_V3": (DeepSeek_V3, "red"),
    "Llama 3 405B": (Llama_3_405B, "blue"),
    "GPT-4": (GPT_4, "green"),
}

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cost/throughput curves")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of samples to use when computing Pareto fronts",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=400,
        help="Resolution of batch size/GPU count search grid",
    )
    args = parser.parse_args()

    scaled_info = {}
    scalings = {}

    iterations_per_model = args.iterations
    overall_progress = tqdm(
        total=iterations_per_model * len(models),
        desc="Overall progress",
    )

    for name, (model, color) in tqdm(models.items(), desc="Models"):
        scaled, factor = scale_to_gpt4o(
            model,
            num_iterations=iterations_per_model,
            grid_size=args.grid_size,
        )
        scalings[name] = (factor, scaled.total_params)
        x, y = curve_for_model(
            scaled,
            name,
            color,
            overall_progress=overall_progress,
            num_iterations=iterations_per_model,
            grid_size=args.grid_size,
        )
        scaled_info[name] = (x, y, color)

    overall_progress.close()

    plot_curves(scaled_info)

    with open("gpt4o_scalings.txt", "w") as f:
        for name, (factor, params) in scalings.items():
            f.write(
                f"{name}: scale factor {factor:.3f}, total params {params/1e9:.2f}B\n"
            )


if __name__ == "__main__":
    main()
