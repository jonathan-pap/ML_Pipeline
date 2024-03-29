# Automated Machine Learning Pipeline Class

- [Session 1 - From Modelling to Production](#session1)
- [Session 2 - Software Engineering for ML](#session2)
- [Session 3 - Toolkit: Git](#session3)
- [Session 4 - Toolkit: Colab & Python](#session4)
- [Session 5 - Toolkit: Python Environments & Visual Studio & Health Infomatics](#session5)
- [Session 6 - Data Set, Data Sample, Data Issues](#session6)
- [Session 7 - Create Fake Data (is Fun!)](#session7)
- [Session 8 - Linear Regression with Fake Data](#session8)
- [Session 9 - Deeplearning Scenario, intro to tensorflow, data Preparation](#session9)
- [Session 10 - Data Augmenting](#session10)
- [Session 11 - Data Balancing](#session11)
- [Session 12 - Data balancing & training effect](#session12)
- [Session 13 - Collecting Data From Storage SQL](#session13)
- [Session 14 - SQL & Python](#session14)
- [Session 15 - Fake Data Creation Part 2](#session15)
- [Session 16 - Data Cleansing.p1](#session16)
- [Session 17 - Data Cleansing.p2](#session17)
- [Session 18 - Data Cleansing.p3](#session18)
- [Session 19 - Data Cleansing.p4](#session19)
- [Session 20 - Tensorflow Data Validation](#session20)
- [Session 21 - Scaling](#session21)
- [Session 22 - Scaling Part 2](#session22)
- [Session 23 - Quantile Transformer](#session23)
- [Session 24 - Recap - scaling](#session24)
- [Session 25 - Linear Regression Tool Kit](#session25)
- [Session 26 - Python Classes](#session26)
- [Session 27 - Coefficients](#session27)
- [Session 28 - Feature Engineering](#session29)

---

<a id=session1></a>
### Session 1 - From Modelling to Production
Intro to ML modelling

- DS Modelling
- DS Life Cycle
- DS principles
- ML pipeline
- Production of ML - ML development
- Production of ML - tasks for apply
- ML Pipeline - Target
- Directed Acyclic Graph
- ML Pipeline - Production ML Infrastructure
- Orchestration References:

  - Executive Data Science: A Guide to Training and Managing the Best Data Scientists (Brian Caffo, Jeff Leek, Roger Peng)
  - The Practical Guide to Managing Data Science at Scale (Domino)
  - Executive Data Science: Coursera-Johns Hopkins University
  - Building Machine Learning Pipelines by Hannes Hapke, Catherine Nelson


:diamond_shape_with_a_dot_inside: [**From Modelling to Production Video**](https://www.youtube.com/watch?v=qFJNkuBRytY)

---

<a id=session2></a>
### Session 2 - Software Engineering for ML

- Application Life Cycle
  - Software Development Life Cycle
-  Data Science Life Cycle

:diamond_shape_with_a_dot_inside: [**Software Engineering for ML Video**](https://www.youtube.com/watch?v=ARUyqRHupMc)

---

<a id=session3></a>
### Session 3 - Toolkit: Git

- Github
- Gitbash

:diamond_shape_with_a_dot_inside: [**Toolkit Git Video**](https://www.youtube.com/watch?v=IcTj5sek30g)

---

<a id=session4></a>
### Session 4 - Toolkit: Colab & Python

- Google Colab
- Install Python

:diamond_shape_with_a_dot_inside: [**Toolkit: Colab & Python Video**](https://www.youtube.com/watch?v=qOHHVdSA8qk)

---

<a id=session5></a>
### Session 5 Toolkit: Python Environments
In this session Thom Ives will explain how to build python virtual environment ... 

- Python 3.x
- Virtual environment wrapper
- System Variables
- Health Informatics Intro *(starts 36:14)*

:diamond_shape_with_a_dot_inside: [**Toolkit: Python Environments & Health Informatics Intro (starts 36:14) Video**](https://www.youtube.com/watch?v=qOHHVdSA8qk)

Ghaith Sankari will show one example about integrating Python project with .net core web api project using vitual studio.

[**VS Video 1**](https://www.youtube.com/watch?v=2dLjHUJ3lZE) | [**VS Video 2**](https://www.youtube.com/watch?v=IZOVSFwIpGo) | [**VS Video 3**](https://www.youtube.com/watch?v=BM3e0p0Iv7w)

---

<a id=session6></a>
### Session 6: Data Set, Data Sample, Data Issues
What is the importance of Data in ML process, what is the sampling and why issues might appears and what is the most important issues

- Feature Space
- Data Samples
- Data Issues
- Data Drift
- Concept Drift

**Assignment**: just explanation based: You take random samples of the same size from a large population and compute the mean of those samples and distribute those samples,
what will form from that distribution?

[Central Limit Theorem](https://www.statisticshowto.com/probability-and-statistics/normal-distributions/central-limit-theorem-definition-examples/)

**Resource**: [ML Data and Concept Drift](https://towardsdatascience.com/machine-learning-in-production-why-you-should-care-about-data-and-concept-drift-d96d0bc907fb)


:diamond_shape_with_a_dot_inside: [**Data Issues Video**](https://www.youtube.com/watch?v=FBivOf73kvw?t=67)

---

<a id=session7></a>
### Session 7 - Create Fake Data (is Fun!)
How to create fake data with Python.

**Assignment**: what is heteroskedasticity. Why is it a challenge, illustrate in notebook.
  - Send DM to Thom, correct answers can share with group.


`import matplotlib.pyplot as plt`<br>
`import random`<br>

`X = [x/10.0 for x in range(100)]` <br>

`Y = [2.0 * x + (random.random() - 0.5) * 0 + 5 for x in X]`<br>

`plt.scatter(X, Y)` <br>
``plt.title('This Is The Title')``<br>
``plt.xlabel('These Are The X Values')``<br>
``plt.ylabel('These Are The Y Values')``<br>
`plt.show()`<br>


added Colab Workbook for heterskedasticity [here](assignment_heteroskedasticity.ipynb)

:diamond_shape_with_a_dot_inside: [**Fake Data is Fun Video**](https://youtu.be/wfc4tNt8ZY8?t=137)

---

<a id=session8></a>
### Session 8 - Linear Regression with Fake Data

**Assignment** Play with the models, ❗ (Please repull the repo)

 1. First run the Fake Data Creations .py.
    1. Fake_Single_Feature_Linear_Data.py
    2. Fake_Single_Feature_NonLinear_Data.py
    3. Fake_Double_Feature_Linear_Data.py
    4. Fake_Double_Feature_NonLinear_Data_with_Functional_Noise.py
 2. Thise will create 5 different .csv files of data
 3. Next run each of the files, in the folder *Intro_to_Regression_Modeling*  and explore and play and understand the functionality of the script. look at the fake data creation.
 
 💡 you can `import sys`, and enter the follow code `sys.quit()` in the script to force stop, so you not running the complete script.
 

 
 4. General_Toolls.py: this file is a module that you can call from with your scirpt, has function to calculate:
    1. print('Mean Square Error       --> [MSE](https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/mean-squared-error/)
    2. print('Root Mean Square Error  --> [RMSE](https://www.statisticshowto.com/probability-and-statistics/regression-analysis/rmse-root-mean-square-error/)
    3. print('Mean Absolute Error     --> [MAE](https://www.statisticshowto.com/absolute-error/)
    4. print('Median Absolute Error   --> [MeDAE](https://www.statisticshowto.com/median-absolute-deviation/)
    5. print('R^2                     --> [r2](https://statisticsbyjim.com/regression/interpret-r-squared-regression/)
    6. print('Adjusted R^2            --> [r2_adj](https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/adjusted-r2/)
  

[Regression Analysis](https://statisticsbyjim.com/regression/when-use-regression-analysis/)

[Regression Statistics](https://www.statisticshowto.com/probability-and-statistics/regression-analysis/)


:diamond_shape_with_a_dot_inside: [**Linear Regression with Fake Data Video **](https://youtu.be/vr1tFEtlv9A)

---

<a id=session9></a>
### Session 9 - Deeplearning Scenario, intro to tensorflow, data Preparation
#### Convolutional neural networks (CNN)

##### Summary of session

- Convolutional Layer
- Effect of Filter Size (Kernel Size)
- Max Pool
- Average Pool
- Batch Sizing
- Padding
- Epochs


👇 Here are some links that have some visual explanations and a playground to experiement.


- [Tinker With a Neural Network](https://playground.tensorflow.org/#activation=tanh&batchSize=6&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1,1&seed=0.77132&showTestData=false&discretize=false&percTrainData=50&x=true&y=false&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
- [Convolution visualizer](https://ezyang.github.io/convolution-visualizer/index.html)
- [What is a Neural Network](https://towardsdatascience.com/what-is-a-neural-network-6010edabde2b)
- [Convolutional Nearual Network Python](https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python)
- [Convolutional Neural Networks cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks?fbclid=IwAR0qn8GjS7fUjexad0GCjXYchU3kqHZOQC3n1Z1bhCw9hFDy3X_FxFoev_8)
- [Convolutional Neural Networks (CNNs) explained Video](https://www.youtube.com/watch?v=YRhxdVk_sIs)
- [A deeper understanding of NNets (Part 1) — CNNs](https://towardsdatascience.com/a-deeper-understanding-of-nnets-part-1-cnns-263a6e3ac61)
- [Difference Between a Batch and an Epoch in a Neural Network](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
- [Epoch vs Iterations vs Batch size](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)
- [Padding and Stride](https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/)
- [Avinash repo; opencv tutorials](https://github.com/AVI18794/OPENCV_TUTORIALS)



**RegeX** : Regular Expression `import re`  [What is regex?](https://regexone.com/)


:diamond_shape_with_a_dot_inside: [**Deeplearning Scenario, intro to tensorflow, data Preparation Video**](https://www.youtube.com/watch?v=_4sjZBacaDM)

---

<a id=session10></a>
### Session 10 - Data Augmenting

**Data Augmenting Techniques**

- Mirroring
  - Flip Horizontal / Vertical
  - Flip Random
  
- Cropping
- Rotate
- Recolor

- PCA, Principal Component Analysis (topic for later lesson)

*Ghaith*

Here is some notes about data augmenting session: 


Data augmentation techniques used in deep learning, but it is still part of data preparation. according to this fact, data augmentation mechanisms will be customized to create important part of ML pipeline.

I wanted to start with data quantity issues solving then we will back to the more fancy and funny part related to data quality.
the assignment for next week is answering the following questions:

- how to perform customized rotation(any value of degree not only 90), code in python is required, and i wish to find presenting volunteers, this task can be performed is many ways and cooperation with other family members to cover many ways to solve the assignment is allowed and appreciated.

- is it possible to re-color the grayscale image, and how: for this question we are not looking for coding examples, we are just looking for explaining and proofing about the answer, you can consider as research task, also brave presenter are highly appreciated.


*[tensorflow data augmentation tutorial](https://www.tensorflow.org/tutorials/images/data_augmentation)*

:diamond_shape_with_a_dot_inside: [**Data Augmenting Video**](https://youtu.be/qwsQvRy5aDI?t=127)

---

<a id=session11></a>
### Session 11 - Data Imbalance

*A classification data set with skewed class proportions is called imbalanced. 
Classes that make up a large proportion of the data set are called majority classes. 
Those that make up a smaller proportion are minority classes*


- created imbalannced data set, by taking sample from each set infected, uninfected
  - using `.sample` from Pands library
 
- Data set, *Name of image, folder name, label*
  - label: 1 = infected, 0 = uninfected

*[monai](https://monai.io/)* good to use on medical sets , with predefined tools for 2D, 3D images.
explanition in video at 14:00 minutes.

- Example of batches sizes 10 showing the class imbalancing.

**Assignment**

To see the impact of oversampling, how the distribution of Data will change.
- Experiment with Batch Sizes
- Experiment with import sample sizes
- What other techniques are there to solve imbalancing with changing number of sample in each batch


Always good to see other tools and share our findings in our **pipeline_class_chat**

MORE TOGETHER!
 
:diamond_shape_with_a_dot_inside: [**Notebook**](https://colab.research.google.com/drive/1s3YNvlWmtwdZhv1beI2UIRO5Q-q_raoh?usp=sharing)
:diamond_shape_with_a_dot_inside: [**Data Imbalance Video**](https://youtu.be/7Xbekc2qMPc)

---

<a id=session12></a>
### Session 12 - Data balancing & training effect

- Weight Computation for Oversmapling & Penalization
- Use of Pre-Trained Models. 
  - Trained on label samples
  - Image net (1million images, split into 1000 catergoires)
  - uses of Resnet18, There are others and different varieties can be used.
- Training and Validation
  - train using `randflipd, randrotae90d, RandGassuanNoised`
  - validation, no transformations
- Training Vs Test Accuarcy


Confusion Matrix

|               | Positive (1) | Negative (0) |
|---------------|--------------|--------------|
| Postitive (1) |      TP      |      FP      |
| Negative (0)  |      FN      |      TN      |

*True Positive, True Negative, False Positive: (Type 1 Error), False Negative: (Type 2 Error)

- Recall = TP / (TP + FN)
- Precision  = TP / (TP+FP)
- F-Score  = 2* Recall * Precision / Recall + Precision (used to compate models)




:diamond_shape_with_a_dot_inside: [**Data balancing & training effect Video**](https://youtu.be/dSDNci03xPg)

---

<a id=session13></a>
### Session 13 - Collecting Data From Storage

Creating Data with SQL, Microsoft SQL Server Managment Studio (*SSMS*)

- Collection of data for Timeseries Analysis
- Randomize data collection
- Using `While` < 10000 to collect 10000 samples
- Using Date to randomize patient transactions for collection
- Create a Procedure that can be called for example in Python,
- Example of creating the ERD (Entity Relationship Diagram, in SSMS

:diamond_shape_with_a_dot_inside: [**Collecting Data From Storage Video**](https://youtu.be/dSDNci03xPg)

--- 

<a id=session14></a>
### Session 14 - SQL & Python

- sqlalchemy
- sqlalchemy engine
- Define functions for server and db connection
- Functions for 
  - Checking table exists
  - Create_table
  - Drop_table
  - Insert Dataframes as Table
  - Update DB
  
 Examples of:
  - SQL query pull and convert to Pandas DF.
  - Pandas DF to SQL Table.
  - Checksum, for detecting errors
  
:diamond_shape_with_a_dot_inside: [**More on Data SQL & Python Video**](https://www.youtube.com/watch?v=1NB1iVj5i0I)

--- 

<a id=session15></a>
### Session 15 - Fake Data Creation Part 2

- Reference to [**Khuyen Tran**](https://www.linkedin.com/in/khuyen-tran-1401/), [`Faker Article`](https://mathdatasimplified.com/2021/05/14/faker-create-fake-data-in-one-line-of-code)
 

- Fake Data for Regression.
  -  Functions to define featuers / Noise / Model
  -  Plotting Model

- Create Fake Classification Data
  -  Functions for Clusters, and Labels
  -  Plott Model

Libraries: `pandas, numpy, json, matplotlib.pyplot`


:diamond_shape_with_a_dot_inside: [**Fake Data Creation Part 2**](https://youtu.be/WZEt9Zks68A)


---

<a id=session16></a>
### Session 16 - Data Cleansing (DF completeness)

- Example in Visuald Studio 19
- Data Cleansing, with Pandas & Numpy
- Resuable Code
- Class and Functions
- Find an Index Column (Unique / Non-Unique)
- Quickly find NaNs and % of Nans per columns
- Find a drop Columns with only 1 unqiue value, (non repeating)


:diamond_shape_with_a_dot_inside: [**Data Cleansing Part 1**](https://youtu.be/EdzxERBVQS0)

---

<a id=session17></a>
### Session 17 - Data Cleansing (Formatting)

Using Functions to determine what datatype a column be in dataframe can be.

**Assignment:** 
- Improvement of code,
- Generalise datetime conversion
- How to process NaN value

Note bookk examples: 
https://github.com/nishamathi/KT/blob/main/DataTypeChk.py

---

:diamond_shape_with_a_dot_inside: [**Data Formatting**](https://www.youtube.com/watch?v=t5h-VlBmatQ)

<a id=session18></a>
### Session 17 - Data Cleansing (Missing Values)

- Lesson discussion on how to impute missing data.
- Mean / Median
- Model prediction to fill out missng data


:diamond_shape_with_a_dot_inside: [**Data Imputation**](https://youtu.be/S6UI9dJAiqc)

---

<a id=session19></a>
### Session 19 - Data Cleansing (Missing Values)
- MCAR: Missing Completely at Random
- MAR: Missing at Random
- MNAR: Missing not at Random
- EDA Correlations
  -   Example with Pearsons Correlation
  
- How to handle differnt types of missing Data

**Assignment**
- Write function that will take x,y and calculation Pearsons Correlation
- Write function that will take x,y and calculation Spearmans Correlation

:diamond_shape_with_a_dot_inside: [**We Care**](https://www.youtube.com/watch?v=k1QVPZLxgkA&t=1541s)

---

<a id=session20></a>
### Session 20 - Tensorflow Data Validation

- Data and Concept Drifts

- Schema Skew
- Feature Skew
- Distribution Skew
- Generate Statistics. 

- If missing a large percentage of data from a feature, get away from the chair, data validation is not about 100% screen time
- Check with the Business Analyst if the feature is required.
- Should there be values for the feature, go back to the Data Engineer understand why these are missing, get them added.

:diamond_shape_with_a_dot_inside: [**Tensorflow Data Validation**](https://www.youtube.com/watch?v=N8VCryrTF_4)

---
<a id=session21></a>
### Session 21 - Scaling Data

Scaling data using various techniques with different libraries

- Numpy
- Pandas
- Tensorflow


:diamond_shape_with_a_dot_inside: [**Scaling**](https://www.youtube.com/watch?v=5KqqQ0NmwYI&list=PLl63hsBNTd2HNq1L0z2l8ri3SHZJ6Egq5&index=25)

---

<a id=session22></a>
### Session 22 - Scaling Data Part2

Scaling data using various techniques with pure Pyython code \
and comparing against libraries results

:diamond_shape_with_a_dot_inside: [**Scaling Part2**](https://www.youtube.com/watch?v=7Y27OAtOtHA)

---

<a id=session23></a>
### Session 23 - Quantile Transformer.

- Beta Distribution
- cumulative probability
- Skewed Data Generation
- Quantile transformation of skewed data.

:diamond_shape_with_a_dot_inside: [**Quantile Transformer**](https://www.youtube.com/watch?v=KY2xhi8M6Gc)

---

<a id=session24></a>
### Session 24 - Scaling recap

:diamond_shape_with_a_dot_inside: 

---

<a id=session25></a>
### Session 25 - Quantile Transformer.

- Linear Relationships Between Features and Labels
- Model Coefficients
- Errors
- Homoscedastic Errors with Zero Means
- Observations of Errors
- Errors are normaly distributed

:diamond_shape_with_a_dot_inside: [**Linear Regression Toolkit**](https://www.youtube.com/watch?v=U_f8g_WA_Cw)

---

<a id=session26></a>
### Session 26 - Quantile Transformer.

- summary of using Classes in Python
-

:diamond_shape_with_a_dot_inside: [**Linear Regression Toolkit**]()

---

<a id=session27></a>
### Session 27 - Coefficients

- feature relationships


:diamond_shape_with_a_dot_inside: [**Linear Regression Toolkit**](https://www.youtube.com/watch?v=_wJgz1kx7MA&list=PLl63hsBNTd2HNq1L0z2l8ri3SHZJ6Egq5&index=30)

---

<a id=session28></a>
### Session 28 - Feature Engineering

- Basic Feature Engineering
- Fake Data
- Visualization
- Modeling withhout Feature Engineering
- Linear
- Polynomial Featuers


:diamond_shape_with_a_dot_inside: [**Linear Regression Toolkit**](https://www.youtube.com/watch?v=v_E1jc48yP4)

---

<a id=session29></a>
### Session 29 - Feature Engineering
