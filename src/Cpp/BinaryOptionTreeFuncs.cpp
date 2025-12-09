#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <vector>
#include <boost/math/distributions/skew_normal.hpp> //boost requires -std=c++14


// @brief: Calculates and returns an array of prices from a binary option pricing tree after num_steps.
// @param S0: Starting price of stock.
// @param num_steps: Number of steps in binary option tree.
// @param dt: Size of time step expressed in years.
// @param T: Length of prediction horizon expressed in years.
// @params nu, sigma, gamma: Parameters for OLS prediction fit in MATLAB
// @params location, scale, shape: Parameters for azzilini distribution fit in MATLAB.
// @param curr_Zt: Pointer to space allocated for current price
// @param curr_Bt: Pointer to space allocated for current Bt (Brownian variable)
// @param curr_Ct: Pointer to space allocated for current Ct (Assymetry variable)   
void expandTree(int num_steps, double S0,
                    double dt, double T,
                    double nu, double sigma, double gamma, 
                    double location, double scale, double shape,
                    double* curr_Zt, double* curr_Bt, double* curr_Ct){

    double h =  1/sqrt(T);
    double c = std::sqrt(dt);

    // //debug
    // std::cout << "location = " << location << "\nscale = " << scale << "\nshape = " << shape << "\n";

    boost::math::skew_normal_distribution<double> dist(location, scale, shape);
    
    int s = pow(2,num_steps-1);

    // ptrs created so we can use one array for both current price and previous price
    double* prev_Bt = (curr_Bt + s);
    double* prev_Ct = (curr_Ct + s);

    if (num_steps % 2 == 0){
        prev_Bt = curr_Bt;
        prev_Ct = curr_Ct;

        curr_Bt = (curr_Bt + s);
        curr_Ct = (curr_Ct + s);
    }

    *prev_Bt = 0;
    *prev_Ct = 0;
    
    for (int i = 0; i < num_steps; i++){


        for(int j = 0; j < pow(2,i); j++){

            double pdf_val = boost::math::pdf(dist, h * *(prev_Bt+j));

            // prev_Bt will be overwritten by the curr_Bt. So need to use its value in Ct
            // first, then assign curr_Bt's value
            *(curr_Ct + j*2) = (i > 0) ? *(prev_Ct+j) + pdf_val: 0;
            *(curr_Ct + j*2+1) = (i > 0) ? *(prev_Ct+j) -  pdf_val: 0;

            *(curr_Bt + j*2) = *(prev_Bt+j) + c;
            *(curr_Bt + j*2+1) = *(prev_Bt+j) - c;
                        
        }


        // Swapping pointers for the next step in the in-place calculation 
        double* temp1 = curr_Bt;
        double* temp2 = curr_Ct;

       
        curr_Bt = prev_Bt;
        curr_Ct = prev_Ct;

        prev_Bt = temp1;
        prev_Ct = temp2;
    }

    curr_Bt = prev_Bt;
    curr_Ct = prev_Ct;

    // in-place calculate stock price from random variable Zt
    for (int i = 0; i <  pow(2,num_steps); i++){

        *(curr_Zt + i) = S0 * exp(
                                    nu * (num_steps) * dt +
                                    sigma * *(curr_Bt + i) +
                                    gamma * c * *(curr_Ct + i)
                                );
        
    }
}


/*
@brief Calculates European option price from a list of simulated stock prices at execution time.
@param prices: The stock prices at expiry
@param strike: The strike price of the option
@param p: The probability of success
@param num_steps: The depth of the option pricing tree
@param calc_option_value: If true, the values in @prices are assumes to be stock prices
    and thus, the option value has to be calculated. If false, @prices is assumed to
    be option prices and dont need this pre-calculation .
*/
double collapseTree(double prices[], double strike, double p, int num_steps, bool calc_option_value = true){


    int N = pow(2,num_steps);

    // Calculate European option price for each stock value
    if (calc_option_value){
        for(int i = 0; i < N; i++){
            prices[i] = std::max(prices[i] - strike, 0.0);
        }
    }

    for (int i = num_steps; i > 0; i--){

        for (int j = 0; j < pow(2,i-1); j++){

            prices[j] = p * prices[2*j] + (1-p) * prices[2*j+1] ;

        }

    }
    // returns current price of option which is stored at i = 0
    return prices[0];
}

double priceForecast(int num_steps, double S0,
                    double dt, double T,
                    double nu, double sigma, double gamma, 
                    double location, double scale, double shape,
                    double* Zt, double* Bt, double* Ct,
                    double p){
    
    expandTree(num_steps, S0, dt, T,
                nu, sigma, gamma,
                location, scale, shape,
                Zt, Bt, Ct);

    double strike = 0; // If calculating stock price, strike is a dummy argument
    double price = collapseTree(Zt, strike, p, num_steps, false);

    return price;
}

std::vector<double> pathForecast(int max_steps, double S0,
                    double dt, double T,
                    double nu, double sigma, double gamma,
                    double location, double scale, double shape,
                    double* Zt, double* Bt, double* Ct,
                    double p
                ){

    std::vector<double> price_series(max_steps,0);
    for(int i = 1; i <= max_steps; i++){
        int sz = pow(2,max_steps);
        memset(Zt,0,sz);
        memset(Bt,0,sz);
        memset(Ct,0,sz);

        int num_steps = i;
        expandTree(num_steps, S0, dt, T,
            nu, sigma, gamma,
            location, scale, shape,
            Zt, Bt, Ct);

        double strike = 0; // If calculating stock price, strike is a dummy argument
        double price = collapseTree(Zt, strike, p, num_steps, false);
        price_series[i-1] = price;
    } 
    return price_series;
}

