## Overview

This git repository provides implementation files for research regarding the Cherny-Shiryaev-Yor SDE. This novel SDE is 
derived from the invariance principles provided by the paper, Cherny et al. (2003). The use of such is to expand the 
Binomial Option Pricing Model for expressing a random process as the sum of drift, a Brownian Variable, and a function
on that Brownian variable. The results for this are being tabulated and a paper will be published shortly with a link on 
this page.

## Table of Contents 
- [Data](#Data)
- [Monte Carlo Fit](#Monte-Carlo-Fit)
- [Dependencies](#Dependencies)
- [References](#References)


## Data
Data for this project was downloaded from two sources:  
> [Massive.com](https://massive.com/) (previously Polygon.io)  
> [Yahoo Finance](https://finance.yahoo.com/)

In compliance with the terms of both data providers, this repo does not provide any data downloaded from these sources. 
Instead, Python scripts are given for working with the APIs of both Massive and Yahoo. 

Users that want to avoid paying for a data subsription to Massive are directed to 
[polygon_data.py](/src/Python/polygon_data.py). This includes some simple functions for pulling data in serial
using a free account. Where pulling data from Massive could be avoided (e.g. daily open or close stock prices), data 
from Yahoo is pulled instead. Keep in mind that free Massive accounts are subject to only 5 API calls per minute.

For individuals with access to at least the starter subscription of Massive's option data, the script, 
[polygon_async.py](/src/Python/polygon_async.py) provides asynchronous wrapper functions for pulling option data
using [HTTPX](https://www.python-httpx.org/).

**Note:** Much of the code was created before Massive change their name from Polygon. Thus, references to the data-
provider in the source code will be directed to "Polygon.io"


## Monte Carlo Fit
The software for implementing a Monte Carlo fit and and running a backtest is located in [MCFit.cpp](/src/Cpp/MCFit.cpp).
Input files for the monte carlo fit can be created by using [markup.py](/src/Python/markup.py) which can either pull 
the (daily) data for testing from yahoo or a .pkl file saved from DataFrame output by polgyon_data.py/polygon_async.py.


## Dependencies
- All Python dependencies are located in the [requirements.txt](/src/Python/requirements.txt)
- C++ dependencies:  
    > Boost 1.88 <https://www.boost.org/>  
    > Open MPI 5.0.7 <https://www.open-mpi.org/>


## References
- Cherny, A. S., Shiryaev, A. N., & Yor, M. (2003). Limit Behavior of the “Horizontal-Vertical” Random Walk and Some Extensions of the Donsker-Prokhorov Invariance Principle. Theory of Probability and Its Applications, 47(3), 377–394. https://doi.org/10.1137/S0040585X97979834
- Message Passing Interface Forum. (2025, June). MPI: A message-passing interface standard version 5.0. <https://www.mpi-forum.org/>
