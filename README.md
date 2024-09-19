# Diamond Price Prediction

## Introduction About the Data:

### Problem Statement

Predicting the price for the stone on the basis of the details given in the dataset so it can distinguish between higher profitable stones and lower profitable stones so as to have a better profit share.

### Data Description

There are 193,573 datapoints in or dataset with 10 independent features inclusing id and 1 dependent feature which is price. Three of our features are object/categorical and six are numerical.

There are 10 independent variable:

- __Id__ : Unique identifier of each diamond.
- __Carat__ : Carat weight of the cubic zirconia.
- __Cut__ : Describe the cut quality of the cubic zirconia. Quality is increasing order Fair, Good, Very Good, Premium, Ideal.
- __Color__ : Colour of the cubic zirconia.With D being the best and J the worst.
- __Clarity__ : Cubic zirconia Clarity refers to the absence of the Inclusions and Blemishes. (In order from Best to Worst, FL = flawless, I3= level 3 inclusions) FL, IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, I3
- __Depth__ : The Height of a cubic zirconia, measured from the Culet to the table, divided by its average Girdle Diameter.
- __Table__ : The Width of the cubic zirconia's Table expressed as a Percentage of its Average Diameter.
- __X__ : Length of the cubic zirconia in mm.
- __Y__ : Width of the cubic zirconia in mm.
- __Z__ : Height of the cubic zirconia in mm.

Target Variable

- __Price__ : Price of the diamond.

### Project Folder Structure

The Project is organised as follows:

- reqirement.txt : Lists all the libraries and dependencies which are required to run this project.
- .gitignore : Lists all the files and directories which need to be ignored by git.
- application.py : Runs flasks application responcible for hosting the web application.
- README.md : Displayes the project documentation.
- artifact : It stores the output of each component which will be used as an input by the next component in training pipeline.




### Exploratory Data Analysis

In this step we perform an elaborated data analysis to understand different aspect of the feature present and summarize main characterstics.

#### Basic Stats:

- The average carat size of a diamond in our dataset is 0.79 with a min of .20 and max of 3.5. Also 25% of the diamonds have a carat size of over 1.3.
- The average depth of a diamond is 61.8 with a minimum depth of 52.1 and maximum of 71.6. Also 25% of the diamonds have a depth of over 62.4.
- The average length of a diamond is 5.7mm with a min length of 0mm and a max lenght of 9.65mm. Also 25% of the diamonds have a length of over 6.51mm.
- The average width of a diamond is 5.72mm with a min width of 0mm and a max width of 10.01mm. Also 25% of the diamonds have a width of over 6.51mm.
- The average height of a diamond is 3.53mm with a min height of 0mm and a max height of 31.3mm. Also 25% of the diamonds have a height of over 4.03mm.
- The average price of a diamond is $3696.15 with a min price of $326 and a max price of $18818. Also 25% of the diamonds have a price of over $5408.

#### Insights

- There are no duplicate records in our dataset.
- There are 5 different cut type with Ideal the most frequent one. Others cuts are based on frequency Premium, Very Good, Good, Fair.
- There are 7 different color type with G the most frequesnt one. Other cuts based on frequency are E, F, H, D, I, J.
- There are 8 different clarity type with SI1 the most frequesnt one.
- We can see a high correlation between x,y,z and also with carat. Therefore we can drop these 3 features x,y,z as this will introduce multicollinearity in our model.
- We have some outliers in feature depth and table.

### Training Pipeline:

In this step we designed a training pipeline which includes data ingestion, data transformation and model training.

- **Data Ingestion**: We created a class which will check if artifact folder exists. Then will load the data from the notebook/data folder and save the raw data file in artifact folder. Then it will split the data in train and test set using train_test_split and was saved in the artifact folder. This class will return the datapath of train and test file.

- **Data Transformation**: We created a class for transformation preprocessor, split the features in numerical and categorical. Define the custom ranking for each ordinal feature. Then created numerical and categorical pipeline. Numerical pipeline will ingest numerical featres and will median impute and scale the data. Similarly categorical pipeline will mode impute, ordinal encode with the help of custom rankings defined and sclae the data. Then with the help of ColumnTransform we will apply these transformers on our data and will save the preprocessor as a pickle file for later use. This class will return the train array and test array.

- **Model Training**



