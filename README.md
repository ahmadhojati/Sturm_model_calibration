# Sturm_model_calibration
Here is the python code for updating Sturm's snow density model coefficients based on SNOTEL density data from 2014 to 2023

# Snow Density Prediction with Sturm's Model

This Python script is designed to predict snow density using Sturm's model. The model is trained and tested on a dataset containing snow-related features such as snow density, snow depth, soil moisture, and climate information. The dataset is filtered based on certain criteria, and the Sturm's model is applied to predict snow density.

## Prerequisites

- Python environment with necessary libraries (NumPy, Pandas, TensorFlow, Scikit-Learn)
- Snow-related dataset in CSV format

## Script Overview

1. **Libraries and Functions:**
   - Import required libraries including NumPy, Pandas, TensorFlow, and scikit-learn.
   - Define a function to determine the water year start date based on the year.
   - Define a function to calculate the day of the year (DOY) given the date.

2. **Data Loading:**
   - Load the snow-related dataset from a CSV file.

3. **Data Filtering:**
   - Apply filters to the dataset based on criteria such as snow density range, average temperature, snow depth, and resolution.
   - Add orbit information to the data.

4. **Coordinate Selection:**
   - Find coordinates with at least a specified number of measurements.
   - Filter the dataset based on selected coordinates.

5. **Model Training and Testing:**
   - Split the dataset into training and testing sets.
   - Scale the features using StandardScaler.
   - Train Sturm's model using deep learning techniques (TensorFlow).
   - Evaluate the model on the testing set.

6. **Model Calibration:**
   - Calibrate the model coefficients using the Nelder-Mead optimization algorithm.

7. **Model Evaluation:**
   - Evaluate the model performance using metrics like RMSE (Root Mean Squared Error) and R2 score on both training and testing sets.

8. **Coefficient Interpretation:**
   - Display the calibrated coefficients for the Sturm's model.

## Usage Instructions:

1. Ensure the Python environment is set up with the required libraries.
2. Provide the snow-related dataset in CSV format.
3. Adjust parameters such as climate class, temperature threshold, and orbit.
4. Run the script and analyze the model's performance metrics.

Note: This script assumes Sturm's model coefficients are provided for different climate classes. If needed, coefficients can be calibrated using the provided optimization process.

Feel free to customize the script based on your specific dataset and requirements.
