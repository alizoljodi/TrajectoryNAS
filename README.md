
# Simulated Annealing for Neural Network Optimization

This script uses simulated annealing to optimize neural network configurations. The primary goal is to identify the best hyperparameter settings for a 3D detection model using CenterPoint on the NuScenes dataset.

---

## Setup

1. **Prerequisites**:
   - Python 3.7+
   - PyTorch
   - `simanneal` package
   - Required libraries in the `det3d` framework.

2. **Installation**:
   - Clone the repository.
   - Install dependencies using `pip install -r requirements.txt`.

---

## Usage

1. **Command Line Arguments**:
   - `--config`: Path to the training configuration file.
   - `--work_dir`: Directory for saving logs and models.
   - `--resume_from`: Resume training from a checkpoint.
   - `--seed`: Seed for reproducibility.

   Example:
   ```bash
   python script.py --config configs/centerpoint/nusc_centerpoint_forecast.py --work_dir ./output
   ```

2. **Execution**:
   Run the script, and the annealer will optimize the hyperparameters based on the energy (latency) of the model.

---

## Outputs

1. **Database**:
   - Results are saved in `models.db` under `all_trials` and `bests` tables.

2. **Logs**:
   - Training logs are saved in the specified `--work_dir`.

---

## Notes

- Ensure you have the correct configurations and dataset paths in the configuration file.
- Adjust the annealer's parameters (`Tmax`, `Tmin`, and steps) based on your computational resources and optimization goals.

---

Feel free to reach out if you have any questions or issues!
