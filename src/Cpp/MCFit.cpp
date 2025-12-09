#include <vector>
#include <cmath>
#include <string>
#include <limits>
#include <filesystem>

#include <mpi.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/circular_buffer.hpp>
   
#include "AzzModel.h"
#include "util.h"

int ENTIRE_WORKLOAD_SIZE;


/// @brief Determines the workload size of a single process based on its rank.
/// @param i Process rank 
/// @param world_size Number of processes
/// @param workload Length of elements in datastructure to be distributed
/// @return Length of elements that process 'i' is to sent and complete
int get_workload_size(int i, int world_size, int workload){
    if (workload % world_size == 0){
        return workload / world_size;
    } 
    int n = (i < (workload % world_size) ) ? ceil((double)workload/world_size) : floor((double)workload/world_size);
    return n;
}


/// @brief Splits up a vector evenly between cores (vector size doesnt have to be 
/// divisible by world_size)
/// @param work Vector (representing a workload) to be split between processes
/// @param world_size Number of processes initialized
/// @return Vector of vectors, where each subvector is specific to a single process
std::vector< std::vector< std::vector<double>>> divide_workload(std::vector<std::vector<double>> &work, int world_size){

    int count = 0;
    int L = work.size();
    std::vector< std::vector< std::vector<double>>> result;
    

    for (int i = 0; i < world_size; i++){

        int n = get_workload_size(i, world_size, L);
        std::vector<std::vector<double>> temp(work.begin() + count, work.begin() + count + n);
        // printVec(temp);
        result.push_back(temp);
        count += n;
    }

    return result;
}


/// @brief Sends a std::vector<double> using MPI. MPI requires a buffer, which in many 
///        cases a user can use the address of the variable being sent. However, within
///        a function, the variable is in the stack in memory while MPI expects it to be 
///        in the heap and causes a segfault. So this functions allocates memory on the 
///        heap for buffers expected by MPI.
/// @param vec std::vector<double> being sent.
/// @param root 
/// @param dest Rank of receiving process.
/// @param tag MPI_TAG for message
/// @param comm communicator
void MPI_send_vector(std::vector<double>& vec, int dest, int tag, MPI_Comm comm){

    int n = vec.size();
    int* n_ptr = (int*)malloc(sizeof(int));
    *n_ptr = n;
    MPI_Send(n_ptr, 1, MPI_INT, dest, tag, comm); 

    int buff_size = n * sizeof(double);
    double* send_buffer = (double*)malloc(buff_size);
    memcpy(send_buffer, vec.data(), buff_size);
    MPI_Send(send_buffer, n, MPI_DOUBLE, dest, tag + 1, comm);

    free(n_ptr);
    free(send_buffer);
}


/// @brief Receive analogue of MPI_send_vector.
/// @param vec Address of vector to store results.
/// @param source Rank of receiving process.
/// @param comm MPI Communicator
void MPI_recv_vector(std::vector<double>& vec, int source, MPI_Comm comm){

    int* n_ptr = (int*)malloc(sizeof(int));
    MPI_Recv(n_ptr, 1, MPI_INT, source, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);

    int n = *n_ptr;
    vec.resize(n);
    int buff_size = n * sizeof(double);
    double* recv_buffer = (double*)malloc(buff_size);
    MPI_Recv(recv_buffer, n, MPI_DOUBLE, source, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
    memcpy(vec.data(), recv_buffer, buff_size);

    free(n_ptr);
    free(recv_buffer);
}


/// @brief Scatters a tensor to all process in a passed in communicator. Cant use MPI 
/// defined scatter method because it requires whatever is to be scatter to be a simple 
//// type like an array. 
/// @param tensor tensor to be scattered
/// @param my_rank Rank of the sending process
/// @param comm Communicator for sending message
/// @return 1D vector containing the a flattened 2D vector of parameters to use for Monte- 
/// Carlo optimization
std::vector<double> scatter_tensor(std::vector<std::vector<std::vector<double>>> &tensor, 
                                   int root, 
                                   MPI_Comm comm){
    
    std::vector<double> my_workload;
    int tag = 0;
    for (int i = 0; i < tensor.size(); i++){
        std::vector<double> vec = ModelParams::flatten_param_matrix(tensor[i]);
        
        // #ifdef DEBUG
        //     std::cout << "i = " << i << std::endl;
        //     std::cout << "vec size = " << vec.size() << " ";
        //     printVec(vec);
        // #endif
        
        if (i == root){
            my_workload = vec;
            tag+=2;
        }
        else{
            MPI_send_vector(vec, i, tag, comm);
            tag+=2;
        }
    } 
    return my_workload;
}


std::vector<double> recv_scattered_tensor(int source, MPI_Comm comm){
    std::vector<double> my_workload;
    MPI_recv_vector(my_workload, source, comm);
    return my_workload;
}


/// @brief Distributes combinations of parameters equally between process. (Called 
///        breadthwise distribution as opposed to distributing subtree to processes)
/// @param params ModelParams object to create parameter matrix from.
/// @param root Root or 'Master' process.
/// @param my_rank Rank of calling process.
/// @param world_size Number of process
/// @param comm Communicator, Generally assumed to be MPI_COMM_WORLD
/// @return Matrix of parameters to be calculated by the calling process
std::vector<std::vector<double>> distribute_breadthwise(ModelParams params, int root, int my_rank, int world_size, MPI_Comm comm){
    using tensor = std::vector<std::vector<std::vector<double>>>;
    using mat = std::vector<std::vector<double>>;
    std::vector<double> my_workload;

    if (my_rank == root){
        mat price_matrix = params.create_param_matrix();
        tensor workloads = divide_workload(price_matrix, world_size);
        my_workload = scatter_tensor(workloads, root, comm);
        ENTIRE_WORKLOAD_SIZE = price_matrix.size();
    }
    else{
        my_workload = recv_scattered_tensor(root, comm); 
    }
    
    mat param_mat = ModelParams::unflatten_param_matrix(my_workload, params);
    return param_mat;
}


void gather_breadthwise(const ModelParams params,
                        const std::vector<std::vector<double>>& param_mat, 
                        int root, 
                        int my_rank, 
                        int world_size,
                        MPI_Comm comm, 
                        std::vector<std::vector<double>>* entire_workload = nullptr //only non-null for root (recieving) process
                    ){

    if (my_rank != root){

        std::vector<double> send_vec = ModelParams::flatten_param_matrix(param_mat);
        int tag = my_rank;
        int n = send_vec.size();
        int buffer_size = n * sizeof(double);
        double* send_buffer = (double*)malloc(buffer_size);
        memcpy(send_buffer, send_vec.data(), buffer_size);
        MPI_Send(send_buffer, n, MPI_DOUBLE, root, tag, comm);
        
        free(send_buffer);
    }
    if (my_rank == root){

        std::vector<std::vector<double>>& entire_param_matrix = *entire_workload;
        std::copy(param_mat.begin(), param_mat.end(), entire_param_matrix.begin());
        int sz = param_mat.size();
        int count = sz;

        // The way the workload is distributed all processes have a workload_size within 
        // 1 of each other. Thus we can use a single buffer and just know where the last 
        // element is. (index=0 has the largest size so just start with that)
        int buff_length = sz * param_mat[0].size();
        double* recv_buffer = (double*)malloc(buff_length * sizeof(double));
        std::vector<double> recv_vec(buff_length, 0);

        for (int i = 1; i < world_size; i++){
            int n = get_workload_size(i, world_size, entire_param_matrix.size());
            

            if (n < sz){
                sz = n;
                buff_length = n * param_mat[0].size();
                recv_vec.resize(buff_length); // processes with rank beyong a specific point
                                     // have 1 less row in theyre workload
            }

            MPI_Recv(recv_buffer, buff_length, MPI_DOUBLE, i, i, comm, MPI_STATUS_IGNORE);
            memcpy(recv_vec.data(), recv_buffer, buff_length * sizeof(double));

            std::vector<std::vector<double>> mat_to_add = ModelParams::unflatten_param_matrix(recv_vec, params);
            std::copy(mat_to_add.begin(), mat_to_add.end(), entire_param_matrix.begin() + count);
            count += mat_to_add.size();
        }
    }
}


// void distribute_depth(){

// }


void Bcast_params(int my_rank, int root, MPI_Comm comm, ModelParams& params){
    
    std::vector<double> serial_params;
    int *n_ptr = (int*)malloc(sizeof(int));
    if (my_rank == root){
        serial_params = params.serialize();
        *n_ptr = serial_params.size();
    }
    MPI_Bcast(n_ptr, 1, MPI_INT, root, comm);

    int n = *n_ptr;
    int buff_size = sizeof(double) * n;
    double* buffer = (double*)malloc(buff_size);
    if (my_rank == root){
        memcpy(buffer, serial_params.data(), buff_size);
    }
    MPI_Bcast(buffer, n, MPI_DOUBLE, root, comm);

    if (my_rank != root){
        serial_params.resize(n);
        memcpy(serial_params.data(), buffer, buff_size);
    } 
    if(my_rank != root)
        params = ModelParams(serial_params);

    free(n_ptr);
    free(buffer);
}


struct InputData {
    std::vector<double> price_series;
    std::vector<double> forecast_idxs; // The index in the price series corresponding to
                                       // the start of each option contract
    std::vector<double> forecast_horizons; // Length of each option contract period
    int num_points_for_fit; // Number of points in training set

    InputData() = default; // Default constructor

    InputData(  const std::vector<double>& d1,
                const std::vector<double>& d2,
                const std::vector<double>& d3,
                const int i)

        : price_series(d1)
        ,forecast_idxs(d2) 
        ,forecast_horizons(d3) 
        ,num_points_for_fit(i){
    }

    // Returns the number of poins that are not in the training set.
    int num_points_for_test(){
        int N = forecast_idxs.size() - num_points_for_fit;
        
        // The start of each option contract is assured to be contained in the price series,
        // however the index of the expiration may not be. If it is not, remove the last
        // option contract period from the test set.
        int end_of_last_contract = forecast_idxs.back() + forecast_horizons.back();
        if (end_of_last_contract > price_series.size() - 1)
            N = N - 1;

        return N;
    }

    void print(){
        std::cout << "num_points_for fit = " << num_points_for_fit << std::endl;
        std::cout << "price_series: " ;
        printVec(price_series);
        std::cout << "forcast_idxs: ";
        printVec(forecast_idxs);
        std::cout << "forecast_horizons: ";
        printVec(forecast_horizons);

    }
};

// TODO Need to refactor this like use a function like MPI_send_vector. However, would
// instead need to make a template that allows int and double types
void Bcast_input(int my_rank, int root, MPI_Comm comm, InputData& input){

    int p;
    int f;
    int* p_ptr = (int*)malloc(sizeof(int));
    int* f_ptr = (int*)malloc(sizeof(int));

    if (my_rank == root){
        *p_ptr = input.price_series.size();
        *f_ptr = input.forecast_idxs.size();
    }

    MPI_Bcast(p_ptr, 1, MPI_INT, root, comm);
    MPI_Bcast(f_ptr, 1, MPI_INT, root, comm);

    p = *p_ptr;
    f = *f_ptr;

    if (my_rank != root){
        input.price_series.resize(p);
        input.forecast_idxs.resize(f);
        input.forecast_horizons.resize(f);
    }

    int pbuff_size = sizeof(double) * p;
    double* price_buffer = (double*)malloc(pbuff_size);

    int fbuff_size = sizeof(double) * f;
    double* idx_buffer = (double*)malloc(fbuff_size);
    double* horizon_buffer = (double*)malloc(fbuff_size);
    int* points_for_fit_buff = (int*)malloc(sizeof(int));

    if (my_rank == root){
        memcpy(price_buffer, input.price_series.data(), pbuff_size);
        memcpy(idx_buffer, input.forecast_idxs.data(), fbuff_size);
        memcpy(horizon_buffer, input.forecast_horizons.data(), fbuff_size);
        *points_for_fit_buff = input.num_points_for_fit;
    }
    MPI_Bcast(price_buffer, p, MPI_DOUBLE, root, comm);
    MPI_Bcast(idx_buffer, f, MPI_DOUBLE, root, comm);
    MPI_Bcast(horizon_buffer, f, MPI_DOUBLE, root, comm);
    MPI_Bcast(points_for_fit_buff, 1, MPI_INT, root, comm);

    if (my_rank != root){
        memcpy(input.price_series.data(), price_buffer, pbuff_size);
        memcpy(input.forecast_idxs.data(), idx_buffer, fbuff_size);
        memcpy(input.forecast_horizons.data(), horizon_buffer, fbuff_size);
        input.num_points_for_fit = *points_for_fit_buff;
    }

    free(p_ptr);
    free(f_ptr);
    free(price_buffer);
    free(idx_buffer);
    free(horizon_buffer);
    free(points_for_fit_buff);
}


/// @brief Returns a subset of param_matrix with the lowest errors
/// Theres a pathological case where the minimum is the fist (or withing the first 10) set
/// of parameters in param_mat. Then, less than 10 parameter combinations are returned.
std::vector<std::vector<double>> find_min(std::vector<std::vector<double>>& param_mat){
    int num_cols = param_mat[0].size();
    boost::circular_buffer<std::vector<double>> cbuf(10);
    double min = std::numeric_limits<double>::infinity();

    for (std::vector<double> row : param_mat){
        double cur = row.back();
        if (cur <= min){
            cbuf.push_back(row);
            min = cur;
        }
    }
    std::vector<std::vector<double>> mins(cbuf.size(), std::vector<double>(num_cols,0));
    for (int i = 0; i < cbuf.size(); i++){
        std::copy(cbuf[i].begin(), cbuf[i].end(), mins[i].begin());
    }
    return mins;
}


std::string get_file_path(std::string filename){
    std::filesystem::path dir = "data";
    std::filesystem::path fname = filename;
    std::filesystem::path input_path = dir / filename;
    return input_path.string();
}


/// @brief Reads input from a json with the format
//   {
//     "price_series": [1.0, 0.5, 0.2 ....],
//     "forecast_indexes": [0, 21, 42];
//     "forecast_horizon": [21, 21, 20]
//   }
/// @param filename 
/// @return 2D vector containing the concatenated input fields
InputData read_input(std::string filepath, InputData input){

    namespace pt = boost::property_tree;
    pt::ptree t_root;

    try {
        pt::read_json(filepath, t_root);

        for (const auto& v : t_root.get_child("price_series")) {
            input.price_series.push_back(v.second.get_value<double>());
        }

        for (const auto& v : t_root.get_child("forecast_indexes")) {
            input.forecast_idxs.push_back(v.second.get_value<double>());
        }

        for (const auto& v : t_root.get_child("forecast_horizons")) {
            input.forecast_horizons.push_back(v.second.get_value<double>());
        }

        int num_points = t_root.get<int>("num_points_for_fit");
        input.num_points_for_fit = num_points;
        
    }

    catch (const pt::json_parser::json_parser_error& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;

    } catch (const pt::ptree_error& e) {
        std::cerr << "Property tree error: " << e.what() << std::endl;
    }

    return input;
}


/// @brief Calcules the probability of a return being positive in a window of the input 
///       series.
/// @param input InputData struct containing info about price_series
/// @param starting_index The index to use as the starting index of the window to 
///        calculate the success rate on
/// @return Success rate 
double calc_success_rate(InputData input, int starting_index){
    int N = input.num_points_for_fit;
    int training_set_start = input.forecast_idxs[starting_index];
    int training_set_end = input.forecast_idxs[starting_index + N] + input.forecast_horizons[starting_index + N];
    int series_length = training_set_end - training_set_start;

    int success_count = 0;
    for (int i = 0; i < series_length; i++){
        int j = training_set_start + i;
        if (input.price_series[j+1] > input.price_series[j]){
            success_count++;
        }
    }
    return (double)success_count / series_length;
}


// RMSE is stored in the last row of the parameter matrix. For a back test, between each 
// iteration of the loop, the RMSE need to be zero-ed out.
void clear_RMSE(std::vector<std::vector<double>>& param_mat){
    for (auto& row: param_mat){
        row[row.size()-1] = 0;
    }
}


/// @brief Calculates the RMSE on the set of parameters in the parameter matrix and 
/// returns the parameters and rmse that have the optimum value
/// @param params ModelParams object for model to be trained.
/// @param param_matrix Matrix of parameters (last index of each row is a zero that will
///                     eventually store the RMSE)
/// @param epm_ptr Pointer to the "entire_param_matrix" that the root process uses to 
///                store all of the parameters and their corresponding RMSE's.
/// @param input InputData 
/// @param num_steps The number of steps to use in the binary option tree
/// @param starting_index Beginning index 
/// @param my_rank Rank of the calling process
/// @param root Rank of the root process
/// @param world_size Total number of processes.
/// @param comm MPI communicator (should generally be MPI_COMM_WORLD)
/// @return The row of the parameter matrix that has the optimum value
std::vector<double> train_model(ModelParams params,
                                std::vector<std::vector<double>>& param_matrix,
                                std::vector<std::vector<double>> *epm_ptr,
                                InputData input,
                                int num_steps,
                                bool dynamic_steps,
                                int starting_index,
                                int my_rank, 
                                int root,
                                int world_size, 
                                MPI_Comm comm,
                                std::string method = "pathwise"){

    int N = input.num_points_for_fit;
    double p = calc_success_rate(input, starting_index);
    double decay_factor = 0.8;
    for (int j = 0; j < N; j++){
        int i = j + starting_index;
        int idx = input.forecast_idxs[i];
        int window_size = input.forecast_horizons[i];
        if (dynamic_steps == true){
            num_steps = window_size - 1;
        }
        std::vector<double> window( input.price_series.begin() + idx,
                                    input.price_series.begin() + idx + window_size);
        double S0 = window[0];
        double observed_return = window[window_size-1] / S0;
        double contract_length = params.time_horizon * window_size;
        double decay = pow(decay_factor,N-1-j);

        if (method == "tree"){
            // double p = 1.0/2.0; // Change this to use success in last window
            ModelParams::monte_carlo_sim( 
                    param_matrix, 
                    num_steps, 
                    S0,
                    contract_length, 
                    p,
                    observed_return,
                    window,
                    decay
                );
        }

        else if (method == "pathwise"){
            ModelParams::mc_from_path(
                    param_matrix, 
                    window, 
                    params.time_horizon, 
                    decay
                );
        }
    }

    // reduce operation for RMSE (rmse is stored in last index of each row of param_matrix)
    // for (std::vector<double> row: param_matrix){
    //     double temp = row.back();
    //     temp = temp / N;
    //     row[row.size()-1] = std::sqrt(temp);
        
    // }

    MPI_Barrier(comm);

    gather_breadthwise( params, 
                        param_matrix, 
                        root, 
                        my_rank,
                        world_size, 
                        comm, 
                        epm_ptr
                    );
                    
    if(my_rank == root){
        std::vector<std::vector<double>> mins = find_min(*epm_ptr);
        printMat(mins);
        return mins.back(); //returns copy/ not reference
        clear_RMSE(*epm_ptr);
    }
    clear_RMSE(param_matrix);
    
    
    // The root process is the only process that returns something meaningful. Since the 
    // return type is not void, the non-root processes return a dummy value.
    std::vector<double> dummy_result;
    return dummy_result;
}

// mpirun -n 4 ./fit_model_debug > data/output.txt
int main(int argc, char* argv[]){

    int root = 0;
    int world_size, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    std::cout << "MPI initialized!" << std::endl;

    assert (argc >= 2 && "Expected at least 2 command line arguments");
    std::string input_filename = argv[1];

    int num_steps = 6;  // Number of steps in Binary Option Tree
    bool dynamic_steps = false;
    std::string fit_method = "pathwise";
    if (argc > 2) {
        std::string arg2 = argv[2];
        if (arg2 == "dynamic-steps"){
            dynamic_steps = true;
            if (my_rank == root){
                std::cout << "Running with dynamic steps.";
            }
        }
        if (argc > 3){
            std::string arg3 = argv[3];
            assert((arg3 == "tree" || arg3 == "pathwise") && "fit type needs to be 'tree' or 'pathwise'");
            fit_method = arg3;
            if (my_rank == root){
                std::cout << "Running with fit_method = " << fit_method << std::endl;
            }
        }
    }

    #ifdef DEBUG
        if (my_rank == root)
            std::cout << "DEBUG MODE ON" << std::endl;
    #endif

    InputData input;
    ModelParams params;
    if (my_rank == root){
        // TODO Add kwargs to the command line arguments for filename input
        std::string price_input = get_file_path(input_filename);
        std::cout << "Reading in price input file from " << price_input << std::endl;
        input = read_input(price_input, input);

        std::string param_input = get_file_path("param_input.csv");
        std::cout << "Reading in parameter input file from " << param_input << std::endl;
        params = ModelParams::readParams(param_input);
    }

    # ifdef DEBUG
        if (my_rank == root){
            // printMat(params.get_data());
            // input.print();
            std::cout << "Broadcasting input." << std::endl;
        }
    #endif

    Bcast_input(my_rank, root, MPI_COMM_WORLD, input);
    Bcast_params(my_rank, root, MPI_COMM_WORLD, params);
    MPI_Barrier(MPI_COMM_WORLD);
    
    #ifdef DEBUG
        if (my_rank == 0){
            std::cout << "Distributing breadthwise and calculating..." << std::endl;
        }
    #endif

    std::vector<std::vector<double>> param_matrix = distribute_breadthwise( 
                                                        params, 
                                                        root, 
                                                        my_rank, 
                                                        world_size,
                                                        MPI_COMM_WORLD
                                                    );
    
    #ifdef DEBUG
        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == 1){
            std::cout << "work distributed between processes, starting loop" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);        
    #endif

    // Allocating space for root process to collect results
    std::vector<std::vector<double>>* epm_ptr = nullptr;
    std::vector<std::vector<double>> entire_param_matrix;
    if(my_rank == root){
        entire_param_matrix.resize(
                                    ENTIRE_WORKLOAD_SIZE, 
                                    std::vector<double>(param_matrix[0].size(), 0)
                                );
        epm_ptr = &entire_param_matrix;
    }

    double squared_error = 0;
    std::vector<std::vector<double>> set_of_optima;
    std::vector<std::vector<double>> set_of_inputs;
    int N = input.num_points_for_test();
    int M = input.num_points_for_fit;
    int loop_length = N - M;

    #ifdef DEBUG
        if (my_rank == 0)
            std::cout << "Looplength = " << loop_length << std::endl;
    #endif

    for (int i = 0; i < loop_length; i++){
        std::vector<double> optimum = train_model(  
                                            params, 
                                            param_matrix, 
                                            epm_ptr, 
                                            input, 
                                            num_steps,
                                            dynamic_steps, 
                                            i,
                                            my_rank,
                                            root,
                                            world_size,
                                            MPI_COMM_WORLD,
                                            fit_method  
                                        );
        
        //debug
        MPI_Barrier(MPI_COMM_WORLD);


        if (my_rank == root){
            int current_idx = input.forecast_idxs[i + M];
            double S0 = input.price_series[current_idx];
            int periods_in_contract = input.forecast_horizons[i + M];
            double observed_return = input.price_series[current_idx + periods_in_contract] / S0;
            double contract_length = params.time_horizon * periods_in_contract;
            double p = calc_success_rate(input, i);

            //debug
            // std::cout << "price Sn = " << input.price_series[current_idx + periods_in_contract] << std::endl;
            std::cout << "p = " << p << std::endl;
            std::cout << "observed_return = " << observed_return <<std::endl;
            std::cout << "S0 = " << S0 <<std::endl; 
            std::cout << "squared_error = " << squared_error << std::endl;
            // std::cout << "current_idx = " << current_idx <<std::endl;
            // std::cout << "periods in contract = " << periods_in_contract << std::endl;
            // std::cout << "forecast_horizons = " ;
            // printVec(input.forecast_horizons);

            std::vector<std::vector<double>> model_sim = {optimum};
            set_of_optima.push_back(optimum);
            set_of_inputs.push_back({S0, contract_length, p});
            std::vector<double> dummy_var = {};
            if (dynamic_steps == true){
                num_steps = periods_in_contract - 1;
            }
            ModelParams::monte_carlo_sim( 
                    model_sim, 
                    num_steps, 
                    S0, 
                    contract_length, 
                    p, 
                    observed_return,
                    dummy_var,
                    1
                );

            squared_error += model_sim[0].back();
        }

        #ifdef DEBUG 
            MPI_Barrier(MPI_COMM_WORLD);
            if(my_rank == root)
                std::cout << "i = " << i << std::endl;
        #endif
    }

    if (my_rank == 0){
        int test_set_size = input.forecast_idxs.size() - input.num_points_for_fit;
        //debug
        std::cout << "test_set_size = " << test_set_size << std::endl;
        std::cout << "squared_error = " << squared_error << std::endl;

        double RMSE = sqrt(squared_error / test_set_size);
        std::cout << "The RMSE is " << RMSE << std::endl;
        std::cout << "The set of optimum parameters is " << std::endl;
        printMat(set_of_optima);
        // mat_to_json(set_of_optima,"data/optima.json");
        std::vector<double> prices = ModelParams::calc_prices(set_of_optima, 10, set_of_inputs);
        printVec(prices);
        std::cout << "Program Complete!" << std::endl;
    }

    MPI_Finalize();
}