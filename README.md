# AI-Numpy-Only

## Fashion-MNIST experiments

The NumPy-only Fashion-MNIST trainer downloads and caches the dataset on first
use:

```bash
python Code/Model_q/fashion_mnist.py --activation relu --epochs 10
```

Compare four activations across several architecture shapes:

```bash
python Code/Model_q/compare_activations.py --epochs 5 --train-limit 10000
```

The default architectures are `64`, `128`, `128-64`, `256-128`, and
`256-128-64`. Each number is the neuron count in one hidden layer. Results are
written to `activation_results.csv` and a ranked JSON summary. Customize the
architecture set like this:

```bash
python Code/Model_q/compare_activations.py \
  --activations relu sin \
  --architectures 32 64-32 128-64-32 \
  --output results/experiment.csv
```

The trainer supports `relu`, `sin`, `cos`, and `tanh`. Both scripts use only
NumPy and Python's standard library.
