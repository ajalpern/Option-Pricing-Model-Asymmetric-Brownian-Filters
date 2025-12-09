#ifndef AZZMODEL_H
#define AZZMODEL_H

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

#include "util.h"
#include "BinaryOptionTreeFuncs.h"

class ModelParams{

private:

    int NUM_PARAMS = 6;

    std::vector<std::vector<double>> data_vec;

    std::vector<double>::iterator nu_iter;
    std::vector<double>::iterator sigma_iter;
    std::vector<double>::iterator gamma_iter;
    std::vector<double>::iterator location_iter;
    std::vector<double>::iterator scale_iter;
    std::vector<double>::iterator shape_iter;

    std::map<   int, 
            std::vector<double>::iterator>
            iter_index_map = {  {0, nu_iter},
                                {1, sigma_iter},
                                {2, gamma_iter},
                                {3, location_iter},
                                {4, scale_iter},
                                {5, shape_iter}
                            };

    void iter_init();

    void iter_reset(int i);

    void param_mat_helper(int i, std::vector<double> &temp, std::vector<std::vector<double>> &result, int &count) const;

public:
    double time_horizon;

    const std::vector<double>& nu() const { return data_vec[0]; }
    const std::vector<double>& sigma() const { return data_vec[1]; }
    const std::vector<double>& gamma() const { return data_vec[2]; }
    const std::vector<double>& location() const { return data_vec[3]; }
    const std::vector<double>& scale() const { return data_vec[4]; }
    const std::vector<double>& shape() const { return data_vec[5]; }

    ModelParams(){
        // Default constuctor for constructing empty parameters.
    }


    ModelParams(double time_horizon_in, 
                std::vector<std::vector<double>> data_vec_in)

            : time_horizon(time_horizon_in),
            data_vec(data_vec_in){
        
        iter_init();
    }


    ModelParams(double time_horizon, 
                std::vector<double> nu_in, 
                std::vector<double> sigma_in, 
                std::vector<double> gamma_in, 
                std::vector<double> location_in, 
                std::vector<double> scale_in, 
                std::vector<double> shape_in)

            :time_horizon(time_horizon), 
            data_vec({nu_in, sigma_in, gamma_in, location_in, scale_in, shape_in}){
        
        iter_init();
    }


    ModelParams(double time_horizon_in, std::vector<std::tuple<double,double,int>> tuples){
        
        // OPT: Use std::copy to prevent vector resizing
        std::vector<std::vector<double>> vec;
        for (auto t : tuples){
            vec.push_back(linspace(std::get<0>(t),
                                    std::get<1>(t),
                                    std::get<2>(t)
                                ));
        }

        time_horizon = time_horizon_in;
        data_vec = vec;

        iter_init();
    }


    ModelParams(std::vector<double> serial_obj){
        
        double time_horizon_in = serial_obj[0];
        std::vector<double> sizes(NUM_PARAMS);
        for (int i = 1; i <= NUM_PARAMS; i++){
            sizes[i-1] = (int)serial_obj[i];
        }

        int offset = 1 + NUM_PARAMS;
        std::vector<std::vector<double>> vec;

        for (int i = 0; i < sizes.size(); i++){
            int s = sizes[i];
            std::vector<double> temp_vec(serial_obj.begin() + offset,
                                         serial_obj.begin() + offset + s);
            vec.push_back(temp_vec);
            offset = offset + s;
        }

        this-> time_horizon = time_horizon_in;
        this-> data_vec = vec;

        iter_init();
    }

    static void monte_carlo_sim(   
                std::vector<std::vector<double>>& param_mat, 
                int num_steps,
                double S0,
                double t,
                double p,
                double observed_return,
                std::vector<double> observed_path,
                double decay
            );

    static void mc_from_path(
                std::vector<std::vector<double>>& param_matrix, 
                std::vector<double> tseries,
                double dt,
                double decay
            );
    
    static std::vector<double> calc_prices( 
                std::vector<std::vector<double>>& param_mat, 
                int num_steps, 
                std::vector<std::vector<double>> set_of_inputs
            );

    static std::vector<double> flatten_param_matrix(const std::vector<std::vector<double>> &mat);

    static std::vector<std::vector<double>> unflatten_param_matrix(std::vector<double> &vec, const ModelParams &params);
    
    const std::vector<std::vector<double>> create_param_matrix();

    const std::vector<double> serialize() const;

    bool empty() const;

    bool has_next() const;

    void next(std::vector<double> &vec);

    static ModelParams readParams(std::string fname, int expected_rows = 7);

    const std::vector<std::vector<double>> get_data();
};



#endif //AZZMODEL_H
