# Fuzzy Logic Classification with FCM and ANFIS

Welcome to the Fuzzy Logic Classification project! This repository implements a fuzzy logic-based multi-label classification system using Fuzzy C-Means (FCM) clustering and Adaptive Neuro-Fuzzy Inference System (ANFIS) training. It is designed to evaluate multiple output labels using fuzzy inference systems (FIS) and track performance metrics across training epochs.

---

## Project Structure

This project is organized into modular MATLAB functions:

| Function Name       | Purpose |
|---------------------|---------|
| `fuzzycm`           | Main orchestration function for multi-label classification |
| `evaluate_fis`      | Trains and evaluates a FIS for a single label |
| `fuzzycm_fis`       | Generates a FIS using FCM clustering |
| `train_neuron`      | Tunes the FIS using ANFIS optimization |
| `get_success`       | Computes success rate and count |
| `success_res`       | Aggregates overall success across all labels |

---

## How It Works

1. **Data Preparation**: The input data is split into training and testing sets, with features and labels organized in `shared_data`.

2. **FIS Generation**: For each label, a fuzzy inference system is created using FCM clustering (`fuzzycm_fis`).

3. **Training**: The FIS is tuned using ANFIS over a specified number of epochs (`train_neuron`).

4. **Evaluation**: The trained FIS is evaluated on the test set, and binary predictions are generated (`evaluate_fis`).

5. **Metrics**: Success metrics are computed per label and overall (`get_success`, `success_res`).

---

## Example Usage

```matlab
% Define parameters
epochs = 50;
clusters = 3;

% Prepare shared_data structure with fields:
% - df_train: training dataset
% - df_test: testing dataset
% - labels: number of output labels
% - features: number of input features
% - output: initialized output matrix
% - results: initialized results table

% Run fuzzy classification
shared_data = fuzzycm(epochs, clusters, shared_data);
```

---

## Output

After execution, `shared_data.results` will contain a summary table like:

| Label     | Successful | Success Rate | Total Rules |
|-----------|------------|--------------|-------------|
| FCM       | 85         | 85 %         | 12          |
| FCM       | 90         | 90 %         | 14          |
| OVERALL   | 80         | 80 %         | 12          |

---

## Requirements

- MATLAB R2021a or later
- Fuzzy Logic Toolbox
- Statistics and Machine Learning Toolbox

---

## Data Format

Your dataset should be structured as follows:

- `df_train`: matrix of size `[num_samples x num_features + num_labels]`
- `df_test`: same format as `df_train`
- Labels should be placed at the end columns of the dataset.

---

## Customization

You can tweak the following parameters:

- `epochs`: Number of training epochs for ANFIS
- `clusters`: Number of clusters for FCM
- `threshold`: Decision threshold for binary classification (default is 0.5)

---

## Tips for Clean Execution

- Ensure `shared_data.output` and `shared_data.results` are initialized before calling `fuzzycm`.
- Use consistent feature-label ordering in both training and testing datasets.
- Validate that `labels` and `features` are correctly set.

---

## Contributing

Contributions are welcome! Feel free to fork the repo, submit pull requests, or open issues for bugs and feature requests.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## FAQ

**Q: Can I use this for multi-class classification?**  
A: This implementation is tailored for multi-label binary classification. For multi-class, consider adapting the FIS structure and output encoding.

**Q: What is the role of `shared_data`?**  
A: It acts as a container for all relevant data and results, enabling modular function calls and easy tracking.

**Q: How do I visualize the rules?**  
A: Use `showrule(fis)` to display the fuzzy rules after training.

---

Let me know if you'd like a version tailored for publishing on MATLAB Central or a simplified README for beginners.
