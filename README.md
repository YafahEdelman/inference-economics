# Inference economics of language models

This is the public repository for the "Inference economics of language models" paper. It contains a Jupyter notebook that contains the models presented in the paper and that you can use to reproduce the paper's results.

The `inference_economics_notebook.py` script now includes a helper function
`scale_model_for_cost_bandwidth` which searches for a scaling of a base model so
that a provided cost and throughput pair lies on the model's cost-throughput
frontier.  The function can be imported and used directly in other scripts.
