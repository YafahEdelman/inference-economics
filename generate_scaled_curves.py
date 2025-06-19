from scaled_curve_helpers import scale_to_gpt4o, curve_for_model, plot_curves
from inference_economics_notebook import DeepSeek_V3, Llama_3_405B, GPT_4

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

scaled_info = {}
scalings = {}

# ``pareto_fronts`` iterates over 1000 samples for each model. Use this value
# to build an overall progress bar across all models.
TOTAL_ITERATIONS_PER_MODEL = 1000
overall_progress = tqdm(
    total=TOTAL_ITERATIONS_PER_MODEL * len(models),
    desc="Overall progress",
)

for name, (model, color) in tqdm(models.items(), desc="Models"):
    scaled, factor = scale_to_gpt4o(model)
    scalings[name] = (factor, scaled.total_params)
    x, y = curve_for_model(scaled, name, color, overall_progress)
    scaled_info[name] = (x, y, color)

overall_progress.close()

plot_curves(scaled_info)

with open("gpt4o_scalings.txt", "w") as f:
    for name, (factor, params) in scalings.items():
        f.write(f"{name}: scale factor {factor:.3f}, total params {params/1e9:.2f}B\n")
