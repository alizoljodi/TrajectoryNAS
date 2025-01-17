
# TrajectoryNAS: A Neural Architecture Search for Trajectory Prediction

This paper, titled TrajectoryNAS, focuses on leveraging Neural Architecture Search (NAS) to enhance the prediction of future trajectories of objects surrounding autonomous systems, with a particular emphasis on autonomous driving (AD). Predicting these trajectories accurately is critical for ensuring safety, avoiding accidents, and maintaining efficient traffic flow, especially in dynamic and stochastic environments where the actions of vehicles and pedestrians are unpredictable.

The study underscores the importance of trajectory prediction in Simultaneous Localization and Mapping (SLAM), where accurate trajectory forecasts can refine object localization and provide critical information about static and dynamic entities in the environment. By enhancing SLAM with predictive capabilities, autonomous systems can make better-informed decisions, reducing risks in complex scenarios.

TrajectoryNAS explores the utility of both 2D and 3D data in this domain. Optical cameras are highlighted for their strength in classification tasks such as recognizing object types, lane markers, and traffic signs, while their performance in measuring distances and velocities is comparatively weaker. Radars complement cameras by offering robust distance and velocity measurements, whereas LIDAR technology stands out as a preferred representation due to its unparalleled accuracy in distance and velocity estimation. LIDAR’s precision makes it particularly suitable for scene understanding applications in autonomous driving and robotics.

The paper also investigates the representation of 3D data in various formats, including depth images, point clouds, meshes, and volumetric grids, to optimize trajectory prediction. The integration of NAS in this context allows for an automated and efficient search for the best-performing neural architectures tailored to trajectory prediction tasks. By automating the design of predictive models, TrajectoryNAS seeks to maximize accuracy and efficiency, ensuring that AD systems can anticipate and adapt to dynamic environments effectively.

In summary, TrajectoryNAS presents a novel approach to improving trajectory prediction for autonomous vehicles, combining the power of NAS with advanced sensor data to deliver safer and more reliable autonomous driving solutions.
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
