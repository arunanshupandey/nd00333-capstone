# Heart Failure Prediction using AutoML and Hyperdrive

Created as a submission for the final Capstone project for Udacity NanoDegree, this repository contains two ways to solve a classification problem. The dataset , as suggested in the project, was taken from Kaggle. The dataset contains people profile for prediction of heart failures. In the project, we used AutoML feature to auto-featurize the dataset and try multiple optimal classification algorithms as a part of automl notebook. The model finalized by the automl is deployed using AzureML SDK and tested. The other path used SKlearn library and find an optimal solution (hyper parameter values) through HyperDrive. 

## Project Set Up and Installation
The only deviation was to store the CSV file into Git Repo and create a datastore at the run time. 
![image](https://user-images.githubusercontent.com/25560357/123817686-fbfaf280-d915-11eb-8dfb-29166fadc9dc.png)

Rest all the steps are as per the recommendation.

## Dataset

### Overview
Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure. Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies. People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

Following is the list of features: 
age
anaemia
creatinine_phosphokinase
diabetes
ejection_fraction
high_blood_pressure
platelets
serum_creatinine
serum_sodium
sex
smoking
time
DEATH_EVENT


### Task
I am using this dataset to predict if the behavioural risk factors can do an early detection and predict the deaths. The models uses classification algorithms to predict the death while a set of symptoms are given. The target variable is named as "DEATH_EVENT". 

![image](https://user-images.githubusercontent.com/25560357/123821399-fbb02680-d918-11eb-80a0-062e9b0392fb.png)

### Access
I have downloaded the file and stored the CSV file into Git Repo. It creates a datastore at the run time.
File Location : https://www.kaggle.com/andrewmvd/heart-failure-clinical-data or https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
 
![image](https://user-images.githubusercontent.com/25560357/123817686-fbfaf280-d915-11eb-8dfb-29166fadc9dc.png)


## Automated ML
I followed the approach used in previous project and experiments to solve a classofication problem using AutoML. 

1. Used Workspace provided by the lab
2. Compute Instance was already created. I created a Compute Cluster with specifiction given ( "STANDARD_DS3_V2" ) to execute the models.
3. Loaded the dataset and used it in notebook
4. Created AutoML Runs and displayed the progress. 
5. Deployed the best model 
6. Tested the best model for output
7. Deleted the services.

Following screenshots provide more details: 

1. Cluster Creation
![image](https://user-images.githubusercontent.com/25560357/123896181-aeff3680-d97e-11eb-8d2f-d679a63eae36.png)

2. Dataset Load
![image](https://user-images.githubusercontent.com/25560357/123896209-b9213500-d97e-11eb-9b97-b280f8b5c79b.png)

3. AutoML Configs
![image](https://user-images.githubusercontent.com/25560357/123896253-d35b1300-d97e-11eb-9ffd-d784fd87d9db.png)

4. AutoML Runs
![image](https://user-images.githubusercontent.com/25560357/123898056-3a2dfb80-d982-11eb-98c7-25f68c0e8779.png)
![image](https://user-images.githubusercontent.com/25560357/123959443-cb759000-d9cb-11eb-8086-3b282533d8ac.png)


5. Deployment of the model
![image](https://user-images.githubusercontent.com/25560357/123898758-8e85ab00-d983-11eb-8b31-c052e47c6eaa.png)
![image](https://user-images.githubusercontent.com/25560357/123898773-96454f80-d983-11eb-8980-fd5477d1da73.png)


6. Test Results
![image](https://user-images.githubusercontent.com/25560357/123898802-a52c0200-d983-11eb-9c5e-f61cd4878d13.png)
![image](https://user-images.githubusercontent.com/25560357/123898834-b117c400-d983-11eb-9ee9-8c91b36994b2.png)


There were multiple suggestions provided in the discussion forum of Kaggle where users have tried to improve the result. In summary, the model could have been improved by increasing the runs, capturing more data, including more features.

## Hyperparameter Tuning

To solve the classification problem, I went with LogisticRegression. I have taken cue from the experiments we did during the course and the easy to understand logic. The logistic regression works well with classification problems. 

Slightly improving the excercises, I have added one more parameter. Along with C and max_iteration I have also used solver. This addition was done while going through the knowledge articles in udacity. The parameters are given as below: 

RandomParameterSampling(
    {'C': choice(0.01, 0.1, 1, 10, 100), 
     'max_iter' : choice(50,75,100,125,150,175,200), 
     'solver' : choice('liblinear','sag','lbfgs', 'saga')})

I have used the primary metric as "Accuracy" for this problem and I have tried to maximize it.

![image](https://user-images.githubusercontent.com/25560357/123900951-aceda580-d987-11eb-807f-876dc1ab345d.png)


### Results

1. The best performing accuracy was 86%
2. The parameters of the model are: ['--C', '0.1', '--max_iter', '50', '--solver', 'liblinear']

I could increase the number of parameter ranges that I have used. I can even change the method of sampling used for the execution to run faster or slower and find good accurate results.

Following screenshots provide more details: 

1. Hyperparameter choices
![image](https://user-images.githubusercontent.com/25560357/123900951-aceda580-d987-11eb-807f-876dc1ab345d.png)

![image](https://user-images.githubusercontent.com/25560357/123914127-b1708900-d99c-11eb-8d82-f4402f7c5cbb.png)

![image](https://user-images.githubusercontent.com/25560357/123914195-c3522c00-d99c-11eb-9268-5ec3befaacec.png)


2. Best Model Run
![image](https://user-images.githubusercontent.com/25560357/123914264-d82ebf80-d99c-11eb-835b-520c9a209ea2.png)

parameters 
![image](https://user-images.githubusercontent.com/25560357/123959410-c284be80-d9cb-11eb-903f-c405e1648639.png)


## Model Deployment
I have deployed AutoML model. The model is deployed as an end point and can be accessed by quering the endpoint with the given inputs. 
The following screenshots provide more details:

1. Deployed endpoints
![image](https://user-images.githubusercontent.com/25560357/123959577-f95ad480-d9cb-11eb-8239-7c2b39f03f40.png)
![image](https://user-images.githubusercontent.com/25560357/123959658-0bd50e00-d9cc-11eb-8045-8b35d1d945c1.png)
![image](https://user-images.githubusercontent.com/25560357/123973133-1fd33c80-d9d9-11eb-9061-cd4e1b446f9d.png)


2. Validation results
![image](https://user-images.githubusercontent.com/25560357/123973407-5e68f700-d9d9-11eb-8113-b9da8d0dd8d6.png)



## Screen Recording
The link can be found here : 
https://youtu.be/o_h3Qz6aajQ


