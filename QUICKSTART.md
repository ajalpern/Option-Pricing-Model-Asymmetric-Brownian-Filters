## Input Files
### Model Input
A sample model parameter input file is given in [param_input.csv](/src/Cpp/data/param_input.csv) and shown below.  
```console
0.00396825396
-.05,.05,5  
.05,.3,5  
.001,.8,5  
0,0,1  
.01,10,5  
5,15,3
```
The format is as follows: The first line is the size of $\Delta t$ in years for each step of the simulation. Every subsequent specifies the parameter
space to search for a minimum for $\nu, \sigma, \gamma$, location, scale, and shape, respectively. The three values in the line specify the left limit,
right limit, and number of points; similar to the linspace() function in MATLAB.

### Price Series Input
Running markup.py will, by default, pull data from Yahoo, but a DataFrame of 1-D price series with a datetime index can be used as input to df_to_json(). 
The output will be a .json of the form
```console
{
  "price_series": [
      6.32,
      7.34,
      7.38,
      ...
  ],
  "date_series": [
      "20250901",
      "20250902",
      "20250903",
      ...
  ],
  "forecast_indexes": [
      5,
      25,
      46,
      ...
  ],
  "forecast_horizons": [
      20,
      21,
      20,
      ...
  ],
  "num_points": 5
}
```
The script uses [pandas-market-calendars](https://pypi.org/project/pandas-market-calendars/) to find the indexes in the time series that are the
issue dates of an option contract of the desired frequency and expiration; entered as the "forecast_indexes". The length of each contract is also 
calculated and entered as "forecast_horizons". The number of contract periods to use for fitting the model is entered as "num_points".

## Compile and Run
The makefile allows compilation for production and debug environments. Prod uses -02 while debug allows more text output. Using debug will allow 
one to view multiple minima found with MC at each time point in the backtest through NDEBUG statements.
```console
$make fit_model
```
add "_debug" to the target for debugging or for more verbose output.

The parallel MC fit was made for both multicore PCs and HPC clusters. Because the parallel program has a simple and robust MapReduce framework, 
knowledge of system architecture is not necessary.  For running on a multicore UNIX system:
```console
$mpirun -n 4 ./fit_model_debug > data/output.txt
```
For submitting jobs and running on HPC clusters with Slurm schedulers, see your institution's guide page. For TTU, the job submission guide can be 
found [here](https://www.depts.ttu.edu/hpcc/userguides/Job_User_Guide.pdf).

This program can be given an input of any number of cores, n, as long as n<= combinations of parameters in the parameter input file.

## Output 
The program prints the set of optimum parameters found at each point in time, in the same order as the input, with an added column for storing the 
corresponding squared error of the training set. Example output for monthly predictions is given below

```conssole
The RMSE is 0.0474532
The set of optimum parameters is 
[[0.05,0.133333,0.8,0,5.26789,50,6.66087e-14],
[-0.0277778,0.272222,0.0897778,0,0.01,55,4.83374e-08],
[-0.0166667,0.05,0.001,0,10,5,0.00110896],
[0.0166667,0.133333,0.711222,0,3.16474,20,5.54783e-15],
[-0.00555556,0.0777778,0.356111,0,7.89684,45,4.84092e-13],
[-0.0166667,0.05,0.001,0,10,5,2.3296e-05],
[0.00555556,0.05,0.178556,0,9.47421,35,6.96768e-13],
[-0.0166667,0.216667,0.267333,0,8.42263,10,1.93159e-15],
[-0.0166667,0.105556,0.533667,0,1.58737,15,3.32639e-17],
[-0.0166667,0.05,0.001,0,10,5,2.32412e-05],
[0.00555556,0.05,0.001,0,1.58737,5,2.00785e-13],
[0.0166667,0.133333,0.0897778,0,2.63895,15,5.60128e-14],
[-0.00555556,0.05,0.356111,0,7.37105,15,6.44566e-14],
[-0.0277778,0.3,0.8,0,0.535789,5,0.000466931],
[-0.0166667,0.05,0.001,0,10,5,0.00641948],
[-0.0166667,0.105556,0.711222,0,0.535789,55,2.07943e-12],
[-0.0166667,0.05,0.001,0,10,5,6.91509e-06],
[0.0388889,0.05,0.444889,0,0.535789,10,4.40643e-14]]
Program Complete!
```





