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
The make file allows compilation for production and debug environments. Prod uses -02 while debug allows more text output. Using debug will allow 
one to view multiple minima found with MC at each time point in the backtest through NDEBUG statements.
```console
$make fit_model
```
add "_debug" to the target for debugging or more info.

The parallel MC fit was made for both multicore PCs and HPC clusters. Because the parallel program has a simple and robust MapReduce framework, 
knowledge of system architecture is not necessary.  For running on a multicore UNIX system:
```console
$mpirun -n 4 ./fit_model_debug > data/output.txt
```





