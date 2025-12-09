#include <vector>
#include <tuple>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <exception>
#include <map>
#include <cmath>

#include <mpi.h>
#include <boost/math/distributions/skew_normal.hpp>

#include "util.h"
#include "BinaryOptionTreeFuncs.h"
#include "AzzModel.h"


void ModelParams::iter_init(){
    for (auto &pair : iter_index_map){
        int index = pair.first;
        pair.second = data_vec[index].begin();
    }
}


void ModelParams::iter_reset(int i){
    for (auto pair : iter_index_map){
        if (pair.first < i){
            pair.second = data_vec[i].begin();
        }
    }
}


/// @brief Serializes the object for message passing in MPI, then a process can 
///        rebuild the object from the serial form using the constructor above
/// @return Vector containing serial object
const std::vector<double> ModelParams::serialize() const{
    
    std::vector<int> temp_vec(NUM_PARAMS, 0);

    int sum = 0;
    for (int i = 0; i < data_vec.size(); i++){ 
        int s = data_vec[i].size();
        sum += s;
        temp_vec[i] = s;
    }

    std::vector<double> serial_vec(1 + NUM_PARAMS + sum);

    serial_vec[0] = time_horizon;
    int offset = 1;
    for (int s : temp_vec){
        serial_vec[offset] = (double)s;
        offset += 1;
    }

    offset = 1 + NUM_PARAMS;
    for (auto v : data_vec){
        std::copy(v.begin(), v.end(), serial_vec.begin() + offset);
        offset += v.size();
    }

    return serial_vec;
}


/// @brief Calculated the errors of a stock forecasts for a set of parameters for fitting
///        a model via Monte Carlo Optimization
/// @param param_mat Matrix of parameer combinations.
/// @param num_steps Number of steps to use in calculating the binary tree.
/// @param S0 Value of the stock at the start of the period.
/// @param t Time horizon of estimation period (i.e. how long is the option in years).
/// @param p Probability of stock increase.
/// @param observed_return Observed return of the stcok at the end of the period
void ModelParams::monte_carlo_sim(  std::vector<std::vector<double>>& param_mat, 
                                    int num_steps, 
                                    double S0, 
                                    double t, 
                                    double p,
                                    double observed_return,
                                    std::vector<double> observed_path,
                                    double decay){

    int sz = pow(2, num_steps);
    double Zt[sz];
    double Bt[sz];
    double Ct[sz];
    memset(Zt,0,sz);
    memset(Bt,0,sz);
    memset(Ct,0,sz);
    
    
    for (auto& row: param_mat){

        std::vector<double> prices = pathForecast( num_steps, S0, 
                                        t/num_steps, t,
                                        row[0], row[1], row[2], 
                                        row[3], row[4], row[5],
                                        Zt, Bt, Ct,
                                        p
                                    );
        
        if (observed_path.size() > 0){
            double se = 0; //squared error
            for (int i = 0; i < prices.size(); i++){
                // If the two series are different lengths, you need to interpolate
                // This is just order 1 interpolation
                int N = observed_path.size();
                if (prices.size()+1 != N){
                    double number = (double)i/prices.size() * (double)observed_path.size();
                    double integer = floor(number);
                    double decimal = number-integer;
                    double index = static_cast<int>(integer); 
                    
                    double observed_price = (index >= N-1) ?
                                observed_path[N-1] :
                                observed_path[index] * decimal + observed_path[index] * (1.0-decimal);
                    se = pow((observed_price - prices[i])/observed_price, 2);
                }
                else {
                    se = pow(observed_path[i+1] - prices[i]/ observed_path[i+1], 2);
                }
                
                row[6] += se * decay;
                    
            }
        }
        else{
            double expected_return = prices[prices.size()-1]/ S0;
            double se = pow(observed_return - expected_return, 2); //squared error
            row[6] += decay * se; // not taking the mean or square root of error, this will be done later
        }
    }
}

void ModelParams::mc_from_path(
                    std::vector<std::vector<double>>& param_matrix, 
                    std::vector<double> tseries,
                    double dt,
                    double decay
                ){
    
    double S0 = tseries[0];
    double h = 1.0/sqrt(dt*tseries.size());
    int n = tseries.size();
    std::vector<int> epsilon(n-1, 0);
    std::vector<double> dts(n-1,0);
    std::vector<double> Bts(n-1,0);
    int sum = 0;
    for (int i = 0; i < n-1; i++){
        if (tseries[i+1] >= tseries[i]){
            epsilon[i] = 1;
        }
        else{
            epsilon[i] = -1;
        }
        dts[i] = (i+1) * dt;
        sum += epsilon[i];
        Bts[i] = sum;
    }

    double dCt = 0;
    std::vector<double> errors(n-1,0);
    for (auto& params : param_matrix){
        double simulated_exponent = 0.0;
        double simulated_price = 0.0;
        boost::math::skew_normal_distribution<double> dist(params[3],params[4],params[5]);
        for (int i = 0; i < n-1; i++){
            i == 0 ? dCt = 0.0 :  dCt = boost::math::pdf(dist, h*sqrt(dt)*Bts[i-1]) * epsilon[i-1];
            simulated_exponent += params[0]*dts[i] 
                                + params[1]*sqrt(dt)*Bts[i]
                                + params[2]*sqrt(dt)*dCt;

            simulated_price = S0 * exp(simulated_exponent);
            errors[i] = abs((tseries[i+1] - simulated_price)/tseries[i+1]); 
        }
        double mape = 0; //mean absolute percentage error
        for (double e : errors){
            mape += e;
        }
        mape = mape/errors.size();
        params[6] += mape * decay;
    }

}

std::vector<double> ModelParams::calc_prices(
                                std::vector<std::vector<double>>& param_mat, 
                                int num_steps, 
                                std::vector<std::vector<double>> set_of_inputs
                            ){

    int sz = pow(2, num_steps);
    double Zt[sz];
    double Bt[sz];
    double Ct[sz];
    memset(Zt,0,sz);
    memset(Bt,0,sz);
    memset(Ct,0,sz);

    std::vector<double> result;

    for (int i = 0; i < param_mat.size(); i++){
        auto row = param_mat[i];
        double S0 = set_of_inputs[i][0];
        double t = set_of_inputs[i][1];
        double p = set_of_inputs[i][2];

        double price = priceForecast( 
                                num_steps, S0, 
                                t/num_steps, t,
                                row[0], row[1], row[2], 
                                row[3], row[4], row[5],
                                Zt, Bt, Ct,
                                p
                            );
        result.push_back(price);
    }
    return result;
}

/// @brief Recursive helper function for create_price_matrix
void ModelParams::param_mat_helper(int i, std::vector<double> &temp, std::vector<std::vector<double>> &result, int &index) const{

    if (i >= NUM_PARAMS){
        temp.push_back(0); //extra space to store error late
        std::copy(temp.begin(),temp.end(),result[index].begin());
        temp.pop_back();
        index++;
    }
    else{
        for (double var : data_vec[i]){
            temp.push_back(var);
            param_mat_helper(i+1, temp, result, index);
            temp.pop_back();
        }
    }
}


/// @brief Creates a matrix of all the combinations of the parameter values with an extra 
/// index in each row for storing the associated price
const std::vector<std::vector<double>> ModelParams::create_param_matrix(){
    
    int s = 1;
    for (int i = 0; i < NUM_PARAMS; i++){
        s *= data_vec[i].size();
    }

    int index = 0;
    std::vector<std::vector<double>> result(s, std::vector<double>(NUM_PARAMS + 1, 0));
    std::vector<double> temp;
    param_mat_helper(0, temp, result, index);
    return result;

}


std::vector<double> ModelParams::flatten_param_matrix(const std::vector<std::vector<double>> &mat){
    std::vector<double> result = reshape2Dto1D(mat);
    return result;
}


std::vector<std::vector<double>> ModelParams::unflatten_param_matrix(std::vector<double> &vec, const ModelParams &params){
    std::vector<std::vector<double>> result = reshape1DTo2D(vec, params.NUM_PARAMS+1);
    return result;
}


/// @brief Essentially this is just hiding the implementation of this class
/// @return Returns if the data_vec is empty
bool ModelParams::empty() const{
    return data_vec.empty();
}


bool ModelParams::has_next() const{
    if (shape_iter == data_vec[5].end() - 1)
        if (scale_iter == data_vec[4].end() - 1)
            if(location_iter == data_vec[3].end() - 1)
                if(gamma_iter == data_vec[2].end() - 1)
                    if(sigma_iter == data_vec[1].end() - 1)
                        if(nu_iter == data_vec[0].end() - 1)
                            return false;

    return true;
}

/// @brief Iterates through parameter combinations
// For Monte Carlo modeling we want combination of all parameters within our search
/// range. For parameter range of 10 points with 7 parameters, this would be 10^7.
/// Instead of storing this, use iterators to iterate through the combinations and
/// the requred storage is only 70 numbers.
/// @param vec Vector for placing the parameter combinations in.
void ModelParams::next(std::vector<double> &vec) {
    
    vec[0] = *nu_iter;
    vec[1] = *sigma_iter;
    vec[2] = *gamma_iter;
    vec[3] = *location_iter;
    vec[4] = *scale_iter;
    vec[5] = *shape_iter;

    for (auto pair : iter_index_map){
        int i = pair.first;
        auto iter = pair.second;

        if (iter != data_vec[i].end()-1){
            iter ++;
            iter_reset(i);
            return;
        }
    }
}

ModelParams ModelParams::readParams(std::string fname, int expected_rows){

    double time_horizon;
    std::vector<std::tuple<double,double,int>> result;
    std::ifstream inputFile(fname);
    
    // It is not recommended to throw user defined errors within MPI programs and instead 
    // one should call MPI_Abort. Thus we return an empty object and check emptiness 
    // within the MPI main function for calling Abort there.
    if (!inputFile.is_open()){
        std::cerr << "Error Opening requested file!" << std::endl;
        return ModelParams();
    }

    std::string line;
    std::getline(inputFile, line);
    time_horizon = std::stod(line);

    while (std::getline(inputFile, line)){
        std::stringstream ss(line);
        std::string value;
        double val1,val2;
        int val3;           
        
        std::getline(ss, value, ',');
        val1 = std::stod(value);
        std::getline(ss, value, ',');
        val2 = std::stod(value);
        std::getline(ss, value);
        val3 = std::stoi(value);

        std::tuple<double,double,int> t(val1, val2, val3);
        result.push_back(t);
    }

    if (result.size()+1 != expected_rows){
        std::cerr << "An unexpected number of lines was in your input parameter csv \n"
            << "Expecting " << expected_rows << ", Received " << result.size() << "\n"
            << "Check format and try again." << std::endl;
        return ModelParams();
    }

    return ModelParams(time_horizon, result);
}

//really this is just for debugging
const std::vector<std::vector<double>> ModelParams::get_data(){
    return data_vec;
}


