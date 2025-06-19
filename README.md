# Inference economics of language models

This is the public repository for the "Inference economics of language models" paper. It contains a Jupyter notebook that contains the models presented in the paper and that you can use to reproduce the paper's results.

The `inference_economics_notebook.py` script now includes a helper function
`scale_model_for_cost_bandwidth` which searches for a scaling of a base model so
that a provided cost and throughput pair lies on the model's cost-throughput
frontier.  The function can be imported and used directly in other scripts.

Additional helper scripts:
- `scaled_curve_helpers.py` provides utilities to scale a model so that it matches a desired cost/throughput pair and to plot cost-throughput curves.
- `generate_scaled_curves.py` demonstrates how to scale DeepSeek_V3, Llama 3 405B and GPT-4 to pass through the GPT‑4o API point (110 tok/s, 8 USD per million tokens) and writes the resulting scalings to `gpt4o_scalings.txt`.
