# Cybersecurity Threat Classification

PyTorch binary-classification project for detecting suspicious cybersecurity
activity from labelled process data.

The work is documented in [`train.ipynb`](./train.ipynb).

## Overview

The model predicts the `sus_label` target:

- `0`: non-suspicious
- `1`: suspicious

The notebook combines the three labelled datasets into a total of `1,141,078`
rows and checks for missing values, duplicates, feature distributions, and
class balance.

## Dataset

The data is divided into three files:

```text
labelled_train.csv
labelled_validation.csv
labelled_test.csv
```

The notebook reports the following shapes:

| Split | Rows | Columns | Purpose |
|---|---:|---:|---|
| Train | 763,144 | 8 | Fit model parameters |
| Validation | 188,967 | 8 | Monitor validation loss |
| Test | 188,967 | 8 | Evaluate the trained model |

The eight columns are seven input features and the `sus_label` target:

- `processId`
- `threadId`
- `parentProcessId`
- `userId`
- `mountNamespace`
- `argsNum`
- `returnValue`
- `sus_label`

The combined dataset is imbalanced:

| Label | Meaning | Count | Proportion |
|---|---|---:|---:|
| `0` | Non-suspicious | 967,564 | 84.7939% |
| `1` | Suspicious | 173,514 | 15.2061% |

Because of this imbalance, accuracy alone is not a sufficient evaluation
metric.

## Workflow

The notebook covers:

1. Mounting Google Drive and loading the three CSV files
2. Inspecting the datasets and their shapes
3. Checking missing values, duplicates, distributions, and class balance
4. Separating the seven features from `sus_label`
5. Standardizing features with `StandardScaler`
6. Fitting the scaler only on the training data, then transforming validation
   and test data with the same scaler
7. Converting the arrays to `float32` PyTorch tensors
8. Creating `TensorDataset` objects and `DataLoader` instances
9. Training a binary neural network while tracking training and validation loss
10. Evaluating the trained model on the test data

The data loaders use a batch size of `64` and shuffle their data.

## Model

The current model is a small multilayer perceptron:

```python
model = nn.Sequential(
    nn.Linear(7, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)
```

Training uses binary cross-entropy with stochastic gradient descent:

```python
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

The training loop is configured with `num_epochs = 10`. It records both
training loss and validation loss for plotting. The notebook output currently
contains loss output for epochs `0` through `19`, so the displayed output and
the current loop configuration are not fully consistent.

Because sigmoid is part of the model, `model(X)` already returns probabilities.
Evaluation code must not apply sigmoid a second time.

## Evaluation

The notebook evaluates the test set with TorchMetrics and reports accuracy,
recall, precision, and a binary confusion matrix.

The saved test evaluation reports:

| Metric | Score |
|---|---:|
| Accuracy | 0.932909 |
| Recall | 0.940056 |
| Precision | 0.985328 |

The confusion matrix is:

```text
[[ 15108   2400]
 [ 10278 161181]]
```

For this matrix, rows represent actual labels and columns represent predicted
labels:

|  | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 15,108 | 2,400 |
| Actual 1 | 10,278 | 161,181 |

The notebook does not currently report F1 score, specificity, or ROC-AUC.

## Dependencies used in the notebook

The notebook imports:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `torch`
- `torchmetrics`
- `google.colab` for Google Drive mounting

## Running the notebook

The notebook is written for Google Colab. It mounts Google Drive and loads the
CSV files from:

```text
/content/drive/MyDrive/Cyber_Security_Threats/
```

Place the three dataset files in that directory, then run
[`train.ipynb`](./train.ipynb) from top to bottom.

## Repository structure

```text
3_cybersecurity threats/
├── labelled_train.csv
├── labelled_validation.csv
├── labelled_test.csv
├── train.ipynb
└── README.md
```
