#ifndef BOTFuncs
#define BOTFuncs

void calc_expected_returns(int num_steps, double S0, 
                    double p, double dt, double T,
                    double nu, double sigma, double gamma, 
                    double location, double scale, double shape,
                    double* curr_Zt, double* curr_Bt, double* curr_Ct);

void expandTree(int num_steps, double S0,
                    double dt, double T,
                    double nu, double sigma, double gamma, 
                    double location, double scale, double shape,
                    double* curr_Zt, double* curr_Bt, double* curr_Ct);

double collapseTree(double prices[], double strike, double p, 
                    int num_steps, bool calc_option_value = true);

double priceForecast(int num_steps, double S0,
                    double dt, double T,
                    double nu, double sigma, double gamma, 
                    double location, double scale, double shape,
                    double* Zt, double* Bt, double* Ct,
                    double p);

std::vector<double> pathForecast(int max_steps, double S0,
                                double dt, double T,
                                double nu, double sigma, double gamma,
                                double location, double scale, double shape,
                                double* Zt, double* Bt, double* Ct,
                                double p);



#endif //BOTFuncs