#include <vector>
#include <ctime>
#include <string>
#include <cassert>
#include <iostream>
#include <fstream>

#include <boost/json.hpp>

#include "util.h"



/// @brief Essentially MATLAB linspace function with slight modification
/// @param start
/// @param end 
/// @param num_points Number of equally spaced points between and including limits
/// @return Vector of equally spaced points between start and end (maybe not including end)
std::vector<double> linspace(double start, double end, int num_points){

    assert(start <= end);
    assert(num_points > 1 || (start == end && num_points == 1));

    std::vector<double> result;

    if (start == end){
        result.push_back(start);
        return result;
    }

    double step_size = (end - start) / (num_points - 1);
    double curr = start;
    while (curr <= end + 1e-14) {
        result.push_back(curr);
        curr = curr + step_size;
    }

    return result;
}


// Returns current time as a string 
// Used as a 'hash' for file creation to prevent overwriting
std::string getTimeString(){
    std::time_t rawtime;
    std::time(&rawtime);

    // Convert time_t to tm structure (local time)
    std::tm* timeinfo = std::localtime(&rawtime);

    // Using strftime to format the time into a C-style string
    char buffer[80]; // Buffer to store the formatted string
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H:%M:%S", timeinfo);
    std::string time_str_c = buffer;
    return time_str_c;
}

/// @brief Flattens a 2D vector to a 1D vector (CAUTION: Assumes the rows in mat are all 
/// the same length)
/// @param mat Input matrix to flatten
/// @return Flattened form of input matrix 
std::vector<double> reshape2Dto1D(const std::vector<std::vector<double>> &mat){
    
    int n = mat.size();
    int m = mat[0].size();
    std::vector<double> vec(n*m,0);

    int offset = 0;
    for (auto row: mat){
        std::copy(row.begin(), row.end(), vec.begin()+offset);
        offset += m;
    }

    return vec;
}

std::vector<std::vector<double>> reshape1DTo2D(const std::vector<double> &input_vec, int num_cols){
    // For portable behavior, it's not smart to throw errors in parallel MPI code.
    // When MPI catched an error it will call Abort() and end the program.
    // This comment is left as a reminder!
    // if (input_vec.size() % num_cols != 0){
    //     throw std::logic_error("Vector Dimensions do not agree. Size of input vector should be divisible by num_cols");
    // }

    int num_rows = input_vec.size() / num_cols;
    std::vector<std::vector<double>> reshaped_vec(num_rows, std::vector<double>(num_cols));
    
    for (int i = 0; i < input_vec.size(); i++){
        reshaped_vec[i/num_cols][i%num_cols] = input_vec[i];
    }

    return reshaped_vec;

}


// Write a 2d Vector to a json file
void mat_to_json(const std::vector<std::vector<double>>& data, std::string filename) {
    boost::json::array json_matrix;

    for (const auto& row : data) {
        boost::json::array json_row;
        for (double element : row) {
            json_row.push_back(element);
        }
        json_matrix.push_back(std::move(json_row));
    }
    
    // Convert the constructed array into the generic JSON value container
    boost::json::value json_val =  boost::json::value_from(json_matrix);
    
    std::ofstream file(filename); 

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    pretty_print(file, json_val);
}


// Taken from the boost website
void pretty_print( std::ostream& os, boost::json::value const& jv, std::string* indent) {
    namespace json = boost::json;
    std::string indent_;
    if(! indent)
        indent = &indent_;
    switch(jv.kind())
    {
    case json::kind::object:
    {
        os << "{\n";
        indent->append(4, ' ');
        auto const& obj = jv.get_object();
        if(! obj.empty())
        {
            auto it = obj.begin();
            for(;;)
            {
                os << *indent << json::serialize(it->key()) << " : ";
                pretty_print(os, it->value(), indent);
                if(++it == obj.end())
                    break;
                os << ",\n";
            }
        }
        os << "\n";
        indent->resize(indent->size() - 4);
        os << *indent << "}";
        break;
    }

    case json::kind::array:
    {
        os << "[\n";
        indent->append(4, ' ');
        auto const& arr = jv.get_array();
        if(! arr.empty())
        {
            auto it = arr.begin();
            for(;;)
            {
                os << *indent;
                pretty_print( os, *it, indent);
                if(++it == arr.end())
                    break;
                os << ",\n";
            }
        }
        os << "\n";
        indent->resize(indent->size() - 4);
        os << *indent << "]";
        break;
    }

    case json::kind::string:
    {
        os << json::serialize(jv.get_string());
        break;
    }

    case json::kind::uint64:
    case json::kind::int64:
    case json::kind::double_:
        os << jv;
        break;

    case json::kind::bool_:
        if(jv.get_bool())
            os << "true";
        else
            os << "false";
        break;

    case json::kind::null:
        os << "null";
        break;
    }

    if(indent->empty())
        os << "\n";
}

