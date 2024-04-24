# Activity Recognition using Temporal Templates

This project demonstrates activity recognition using temporal templates, specifically Motion-History Image (MHI) and Binary Motion-Energy Image (MEI). The goal is to extract features from input datasets, train machine learning models, and predict actions based on the extracted features.

## Getting Started

Follow these steps to get started with the project:

1. **Clone the Repository:**
   o clone this repository to your local machine, use the following command:

```bash
git clone https://github.com/your-username/your-repository.git

3. **Install Dependencies:**
-pip install -r requirements.txt
4. **Dataset Preparation:**
- Download Your Dataset through this link https://www.crcv.ucf.edu/data/UCF_Sports_Action.php

4. **Feature Extraction:**
## Usage

1. **Load Dataset**: 
    - Place your dataset in the appropriate directory.
    - Open a terminal and navigate to the project directory.
    - Run `featureextraction.py` with the dataset path as a command-line argument.

```bash
python featureextraction.py /path/to/your/dataset
.
- Run `featureextraction.py` to extract features from the dataset.
- Features will be saved in an NPZ file named `features.npz`.

5.## Machine Learning Model Training

1. **Open `mlmodel.py`**: 
    - Navigate to the project directory in your terminal.
    - Open `mlmodel.py` in a text editor of your choice.

2. **Load Extracted Features**: 
    - Inside `mlmodel.py`, import the necessary libraries to load data from the NPZ file.
    - Load the extracted features from `features.npz` using the appropriate function (e.g., `numpy.load()`).
    - Extract the features and labels from the NPZ file.

3. **Train the Machine Learning Model**:
    - Choose the appropriate machine learning algorithm for your task (e.g., SVM, RandomForest, etc.).
    - Split the dataset into training and testing sets using functions like `train_test_split()` from scikit-learn.
    - Train your machine learning model using the training data.
    - Fine-tune hyperparameters if necessary to improve model performance.

4. **Save the Trained Model**:
    - After training, save the trained model to a file for later use.
    - You can use libraries like `joblib` or `pickle` to serialize the model object and save it as a file.
    - Provide a descriptive filename for the saved model (e.g., `trained_model.pkl`).

5. **Testing and Evaluation (Optional)**:
    - Optionally, you can evaluate the performance of your trained model using the testing dataset.
    - Make predictions on the testing data using the trained model.
    - Compute evaluation metrics such as accuracy, precision, recall, and F1-score to assess the model's performance.

6. **Usage of the Trained Model**:
    - Once the model is trained and saved, you can use it to make predictions on new data.
    - Load the saved model in your application using appropriate functions (e.g., `joblib.load()`).
    - Use the loaded model to predict actions on new instances of data.

7. **Save Changes and Close the File**:
    - Save the changes made to `mlmodel.py` after training and model saving.
    - Close the file in your text editor.



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


