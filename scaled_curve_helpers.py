import matplotlib.pyplot as plt
from inference_economics_notebook import (
    H100,
    DeepSeek_V3,
    Llama_3_405B,
    GPT_4,
    TokenEconSettings,
    ComparisonSettings,
    pareto_fronts,
    token_latency_seconds_default,
    scale_model_for_cost_bandwidth,
)


def scale_to_gpt4o(model, *, num_iterations=1000, grid_size=400):
    """Scale ``model`` so that the GPT-4.1 price/perf point lies on its frontier.
    Returns the scaled model and scaling factor."""
    scaled = scale_model_for_cost_bandwidth(
        target_cost=8.0,
        target_tokens_per_second=110.0,
        base_model=model,
        gpu=H100,
        num_iterations=num_iterations,
        grid_size=grid_size,
    )
    scale_factor = scaled.total_params / model.total_params
    return scaled, scale_factor


def curve_for_model(
    model,
    name,
    color,
    overall_progress=None,
    num_iterations=1000,
    grid_size=400,
):
    """Return the cost/throughput curve for ``model``.

    If ``overall_progress`` is provided, it will be updated by
    :func:`pareto_fronts` to allow tracking progress across multiple models.
    ``num_iterations`` controls how many samples :func:`pareto_fronts` uses,
    trading off accuracy for runtime. ``grid_size`` sets the resolution of the
    search space for batch size and GPU count.
    """

    settings = [TokenEconSettings(name=name, gpu=H100, model=model, input_len=0, color=color)]
    comp = ComparisonSettings(settings, "tmp", "tmp")
    x, y, _, _, _ = pareto_fronts(
        comp.comparison_list,
        token_latency_seconds_default,
        use_pp=True,
        overall_progress=overall_progress,
        num_iterations=num_iterations,
        grid_size=grid_size,
    )[0]
    return x, y


def plot_curves(curve_data, output_file="gpt4o_scaled_curves.png"):
    plt.figure(figsize=(8, 6))
    for name, (x, y, color) in curve_data.items():
        plt.plot(x, y, label=name, color=color)
    plt.scatter([110], [8], color="black", label="GPT-4.1 API")
    plt.xlabel("Tokens per second per request")
    plt.ylabel("Cost per million tokens (USD)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)

