# Mini-project IV

### [Assignment](assignment.md)

## Project/Goals
Deploy a barebones ML api to AWS.

## Hypothesis
Some groups of people I feel like might have higher loan approval rate:
- Married Males
- Educated from Urban areas
- People with no dependents and aren't self employed

## EDA 
Incomes and loan amounts were heavily right skewed, though interestingly coapplicant income was consistently lower than applicant income. Loan terms were slightly left skewed, and most applicants had a credit history. Most applicants were male, married, had no dependents, were not self-employed, and had graduated from school.


## Process
Used pipelines throughout the following steps.

### Data Cleaning
Imputed null values with medians as to circumvent the effect of skew. 

### Data Wrangling 
Applied a log transform on right skewed financial features.

### Feature Engineering
Merged applicant and coapplicant incomes into total income.

### Model Selection
Chose logistic regression as the dataset was small and the desired output was to predict the probability of a binary categorical label. Here the dataset was split into train and test data.

### Hyperparameter Tuning
Performed cross validated grid search to find an appropriate regularization type and weight.

### Final Model
Using the optimum hyperparameters found, a final model was trained on the entire training set.

## Results/Demo
Model performed at 80% cross validated accuracy and 82% accuracy on test data, possibly indicating a slight underfit. Depending on the random state, these numbers sometimes fluctuated, and so we can't say for certain if it is underfit given the small dataset. However the approximation error seems to be quite low.

This model was then exported with pickle and flask into AWS.

Here is the endpoint:
http://ec2-18-116-60-101.us-east-2.compute.amazonaws.com:5555/scoring

It accepts POST requests with a json body of the form:
```json
{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Not Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 10000,
    "CoapplicantIncome": 10000,
    "LoanAmount": 200,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
}
```
## Challanges 
Numerous technical issues arose when trying to deploy the model into the cloud. I had to remake the instance 4 times.

## Future Goals
On the ML side I would try other models, augment the data, and do more feature engineering.

On the API side I would add more functionality for different models and for more information out of those models.