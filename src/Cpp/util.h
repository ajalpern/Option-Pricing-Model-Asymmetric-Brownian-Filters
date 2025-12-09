#ifndef utilFuncs
#define utilFuncs
#include <vector>
#include <string>
#include <iostream>

#include <boost/json.hpp>


template <typename T>
void printVec(const std::vector<T>& vec, bool endline = true){
    std::cout << '[';

    for (size_t i = 0; i < vec.size(); i++){
        std::cout << vec[i]; //std::cout << is type safe as opposed to sprintf
        
        if (i < vec.size() - 1){
            std::cout << ",";
        }
    }
    std::cout << "]";
    if (endline == true) std::cout << "\n";
}


template <typename T>
void printMat(const std::vector<std::vector<T>>& mat, 
                bool endline = true){
    std::cout << "[";

    for (size_t i = 0; i < mat.size(); i++){
        printVec(mat[i], false);

        if (i < mat.size() - 1){
            std::cout << "," << "\n";
        }
    }
    std::cout << "]";
    if (endline == true) std::cout << "\n";
}


template <typename T>
void print3DMat(const std::vector
                        <std::vector
                            <std::vector<T>>>& mat){
    std::cout << "[";
    for (size_t i = 0; i < mat.size(); i++){
        printMat(mat[i], false);

        if (i < mat.size() - 1){
            std::cout << "," << "\n";
        }
    }
    std::cout << "]" << "\n";
}


std::vector<double> linspace(double start, double end, int num_points);

std::string getTimeString();

std::vector<double> reshape2Dto1D(const std::vector<std::vector<double>> &mat);

std::vector<std::vector<double>> reshape1DTo2D(const std::vector<double> &input_vec, int num_cols);

void mat_to_json(const std::vector<std::vector<double>>& data, std::string filename);

void pretty_print( std::ostream& os, boost::json::value const& jv, std::string* indent = nullptr);


#endif //utilFuncs