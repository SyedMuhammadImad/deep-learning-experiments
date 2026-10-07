# Normalization, regularization and recurrent/vision experiments

A completed implementation collection: explicit NumPy batch/layer normalization and dense backpropagation; scalar Momentum, RMSProp and Adam; framework baseline/L2/dropout models; SimpleRNN/LSTM/GRU sentiment models; and plain/residual image classifiers. Original filled and shared reference layouts are credited in `ATTRIBUTION.md`. Assisted repairs and newly executed comparisons are documented rather than presented as independent research.

## Environments and commands

Use two Python 3.12 environments: install `requirements.txt` for NumPy/PyTorch and `requirements-tensorflow.txt` separately for TensorFlow. The tested NumPy versions differ between environments. Do not install both pinned NumPy versions into the same environment. Set `PYTHONUTF8=1`, `OMP_NUM_THREADS=2`, `TF_NUM_INTEROP_THREADS=2` and `TF_NUM_INTRAOP_THREADS=2` on Windows.

NumPy/PyTorch:

```powershell
python -m pytest -q -p no:cacheprovider test_numpy_layers.py test_vision.py
python numeric_experiment.py --heart-csv local/heart.csv --output-dir local/numeric
python vision_experiment.py train --data-dir local/cifar --output-dir local/vision --epochs 5
python vision_experiment.py predict --model-dir local/vision --architecture residual --image local/input.png
```

The heart CSV requires the retained combined-table columns and `num` target, mapped to zero versus greater than zero. `id` and `dataset` are excluded. Download the official CIFAR-10 Python archive with torchvision into the local data directory before training; the experiment deliberately requires local data. The prediction command reads a bounded local image and prints its label/uncalibrated probabilities. No picture is saved or published.

TensorFlow:

```powershell
python -m pytest -q -p no:cacheprovider test_sequence.py test_regularization.py
python regularization_experiment.py --output-dir local/regularization --epochs 30
python sequence_experiment.py train --output-dir local/sequence --epochs 4
python sequence_experiment.py predict --model local/sequence/lstm.keras --tokens-json local/review-token-ids.json
```

Keras obtains the official pre-indexed IMDB archive and checks its publisher checksum. The token-ID prediction interface uses that encoder's integer vocabulary, not an invented raw-text tokenizer. The input JSON file is an integer list such as `[1,14,22,16,43]`. Vocabulary 10,000 and review length 100 are fixed; padding ID 0 is masked. Raw text/data, model binaries, pictures and credentials remain local. Data/model paths are explicit.

## Fresh results

All neural checkpoints are selected by validation objective, then reloaded before held-out evaluation. No test score selects an epoch. Numeric tables use stratified 60/20/20 row splits with training-only imputation/scaling/encoding. Heart-table site grouping was not available, so these measurements do not establish external/cohort performance. The retained original NumPy model applied normalization in the forward pass but omitted its backward derivative. The repaired implementation trains gamma/beta and propagates through every normalization layer before updating any weights. Every parameter's finite-difference gradient and normalization's independent autograd gradient agree.

| Comparison | Held-out accuracy |
|---|---|
| NumPy breast-cell | plain 95.6%, batch_normalized 94.7% |
| NumPy combined heart | plain 83.7%, batch_normalized 82.6% |
| Dense regularization | baseline 97.4%, l2 96.5%, dropout 97.4% |
| IMDB recurrent | rnn 51.8%, lstm 75.9%, gru 74.7% |
| CIFAR plain/residual | plain 44.9%, residual 40.5% |

Normalization and residual connections did not improve these particular held-out accuracies; those weaker results remain visible. Dense regularization runs have a maximum of 30 epochs on 341 training/114 validation/114 test observations, rather than claiming a historical 300-epoch experiment. L2 validation loss includes its penalty, so cross-model performance comparisons use held-out classification metrics, not the penalized training objectives. Regularization is not universally better. Both numeric datasets are offline benchmarks, with no clinical suitability claim.

Recurrent training uses 4,000 training/800 validation examples drawn from the official training split, followed by 1,000 examples from the official test split. Four epochs is the maximum; early stopping and validation checkpointing are applied. The simple RNN remains weak. The standard pre-indexed vocabulary is a supplied benchmark vocabulary; no claim is made that it was re-fit on this subset.

Image training uses 4,000 training/1,000 validation examples from the official training split, followed by 1,000 balanced official test examples, five epochs and a small 32-channel/three-block architecture. Pixel normalization fits training images only. Horizontal flipping is applied only in the training loop, avoiding the original shared-transform mutation between train/validation subsets. This is a limited CIFAR comparison, not full benchmark performance. The original recursive skip-on-corrupt-image helper is not retained; invalid inference images fail explicitly.

## Verification and limits

Twenty-one checks cover every NumPy parameter gradient, batch/layer normalization input and affine gradients, frozen running statistics, constant inputs, rejected targets without mutation, checkpoint shape/statistic reloads, independent optimizer steps, residual identity, full image-network gradients, malformed images, recurrent mask invariance and save/reload, token bounds, independent L2 penalty gradients and stochastic training/deterministic inference for dropout. Fresh data runs include all four numeric models, three dense regularization models, three recurrent models and two image models. Actual saved image inference was exercised on an official local test image. Five notebooks follow setup → implementation/experiment → results, have executed code cells and omit outputs/media.

This is a curated collection of finished implementations. Original drafts/templates are preserved locally. Historical Cats/Dogs, six-landscape and pretrained-transfer datasets/experiments and an AAPL forecasting draft are not reproduced or assigned invented metrics. One-seed/subset results, source-data validity, unseen operational data and physical hardware are not certified. `metrics.json` records sample counts, histories, confusion matrices, selected epochs, optimizer trajectories and these limits.
