# ML-Group Competition
Entry by Team 46 for Group Income Prediction Kaggle Competition

# Requirements
  * XGBoost
  * Scikit-Learn
  * Numpy
  * Pandas
  * Category Endcoders
  
# Usage
Script is contained in code folder. Data should be placed in its own folder, 'data', located in root of project folder. Relative filepaths are used.

All preprocessing and modelling is in the one file, as well as creation of the output CSV file, ready for submission.
  
# Known Issues/Improvements
  * Overfitting slightly towards the end of competition
  * Hyperparameters could be better tuned using CV
  * Further feature bagging and scaling may improve results
