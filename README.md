# Activity Recognition using Temporal Templates

This project demonstrates activity recognition using temporal templates, specifically Motion-History Image (MHI) and Binary Motion-Energy Image (MEI). The goal is to extract features from input datasets, train machine learning models, and predict actions based on the extracted features.

## Getting Started

Follow these steps to get started with the project:

1. **Clone the Repository:**

2. **Install Dependencies:**

3. **Dataset Preparation:**
- Prepare your dataset for activity recognition. The dataset should contain video sequences or frames of various actions.

4. **Feature Extraction:**
- Open `featureextraction.py` and specify the path to your dataset.
- Run `featureextraction.py` to extract features from the dataset.
- Features will be saved in an NPZ file named `features.npz`.

5. **Machine Learning Model Training:**
- Open `mlmodel.py` and load the extracted features from `features.npz`.
- Train your machine learning model using the extracted features.
- Save the trained model for later use.

6. **Action Prediction:**
- Open `mlmodel.py` and load the trained machine learning model.
- Provide input features to the model to predict actions.

## File Structure

- `featureextraction.py`: Script to extract features from input datasets.
- `mlmodel.py`: Script to train machine learning models and predict actions.
- `requirements.txt`: List of Python dependencies required for the project.

## Contributing

Contributions to this project are welcome! Please follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).


