# Automated Machine Learning Pipeline Class

- [Session 1 - From Modelling to Production](#session1)
- [Session 2 - Software Engineering for ML](#session2)
- [Session 3 - Toolkit: Git](#session3)
- [Session 4 - Toolkit: Colab & Python](#session4)
- [Session 5 - Toolkit: Python Environments & Visual Studio & Health Infomatics](#session5)
- [Session 6 - Data Set, Data Sample, Data Issues](#session6)
- [Session 7 - Create Fake Data (is Fun!)](#session7)

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


`import matplotlib.pyplot as plt`
`import random`

`X = [x/10.0 for x in range(100)]`

`Y = [2.0 * x + (random.random() - 0.5) * 0 + 5 for x in X]`

`plt.scatter(X, Y)
``plt.title('This Is The Title')`
``plt.xlabel('These Are The X Values')``
``plt.ylabel('These Are The Y Values')``
`plt.show()`



:diamond_shape_with_a_dot_inside: [**Fake Data is Fun Video**](https://youtu.be/wfc4tNt8ZY8?t=137)

---


