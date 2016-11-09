/**
*	@file stp_runner.cpp
*	Main file of the stp runner
*	Contains the main function. Creates and configures the TCLAP and Spdlog interface. Calls test functions.
*/

#include "stp_runner.h"

//Function type map
std::map<std::string,func> func_types = {
    {"tophat", tophat},
    {"triangle", triangle},
    {"sinc", sinc},
    {"gaussian", gaussian},
    {"gaussian-sinc", gaussianSinc},
    {"tophat-kernel", kernel_tophat},
    {"triangle-kernel", kernel_triangle}
};

//Logger
std::shared_ptr <spdlog::logger> _logger;

//Selection flag
TCLAP::ValueArg<std::string> _actionFlag("a", "action", "Chooses which function to use\ntophat for tophat\nsinc for sinc\ngaussian for gaussian\ntriangle for triangle\ngaussian-sinc for gaussian-sinc\ntophat-kernel to create a tophat kernel\ntriangle-kernel to create a triangle kernel", true, "tophat", "string");

//Filepath flag
TCLAP::ValueArg<std::string> _fileArg("f","filepath","Input filepath",true,"../mock_uvw_vis.npz","string");

//Key flag
TCLAP::ValueArg<std::string> _keyArg("k","key","Name of the array we wish to extract from an npz file",false, "false", "string");

//Command line parser
TCLAP::CmdLine _cmd("Parser for the stp runner", ' ', "0.3");

/**
*	@brief Creates the switch flags to be used by the parser
*
*	Creates a list of switch arguments and then xor adds to the command line parser, meaning that only
*	one flag and one only is required to run the program_invocation_name
*
*/
void createFlags() throw(TCLAP::ArgException)
{
	//Adds the action flag to the parse. It chooses which function I wish to call from the libstp
	_logger -> debug("Adding the action flag to the command line parser");
	_cmd.add(_actionFlag);

	//Adds the value flag to the parser list. This is flag is required
	_logger -> debug("Adding the filepath flag to the command line parser");
	_cmd.add(_fileArg);

	//Adds the key flag to the parser list. At least one key is required
	_logger -> debug("Adding the key flag to the command line parser");
	_cmd.add(_keyArg);
}

/**
* 	@brief Logger initialization function
*
*	Creates and initializes the logger to be used throughout the program
*/
void initLogger() throw(TCLAP::ArgException)
{
	//Creates two spdlog sinks
	//One sink for the stdout and another for a file
	std::vector <spdlog::sink_ptr> sinks;
	sinks.push_back(std::make_shared <spdlog::sinks::stdout_sink_st> ());
	sinks.push_back(std::make_shared <spdlog::sinks::simple_file_sink_st> ("logfile.txt"));
	_logger = std::make_shared <spdlog::logger> ("logger", begin(sinks), end(sinks));
}

/**
*	@brief Main function of the stp-runner interface
*
*	This main function calls functions from the libstp and runs tests on them as well as
*	retrieve pertinent debug data
*
*	@param[in] argc Number of arguments the program receives
*	@param[in] argv Array of strings containing the arguments of the program
*	
*	@return A code indicating the program exit status
*/
int main(int argc, char** argv)
{
	//Creates and initializes the logger
        initLogger();
	_logger -> info("Program start");

	//Adds the flags to the parser
	createFlags();
	
	//Parses the arguments from the command console
	_logger -> info("Parsing arguments from console");
	_cmd.parse(argc, argv);

	//Creates the input armadillo matrix for the STP functions
	arma::mat input;

	//If a key is present loads an npz, otherwise loads an npy
	if (_keyArg.getValue().compare("false") != 0)
	{
	    cnpy::NpyArray npz_data = cnpy::npz_load(_fileArg.getValue(), _keyArg.getValue());
	    input = arma::trans(loadNpyArray(&npz_data));
	}
	else
	{
	    cnpy::NpyArray npy_data = cnpy::npy_load(_fileArg.getValue());
	    input = arma::trans(loadNpyArray(&npy_data));
	}

	//Creates the output armadillo matrix
	arma::mat output;
        
        //Create variables to use in convolution functions
        const double half_base_width(3.0);
        const double triangle_value(1.0);
        const double gaussian_width(1.0);
        const double width_normalization_gaussian(2.52);
        const double width_normalization_sinc(1.55);
        const int support_tophat(1);
        const int support_triangle(2);
        const double oversampling_tophat(3.0);
        const double oversampling_triangle(1.0);
 
        //Chooses the function to run
	try
        {
            switch(func_types.at(_actionFlag.getValue()))
            {
                case tophat:
                    _logger -> info("Starting Tophat");
                    TopHat tophat;
                    output = tophat(input, half_base_width);
                    break;
                case triangle:
                    _logger -> info("Starting Triangle");
                    Triangle triangle;
                    output = triangle(input, half_base_width, triangle_value);
                    break;
                case sinc:
                    _logger -> info("Starting Sinc");
                    Sinc sinc;
                    output = sinc(input);
                    break;
                case gaussian:
                    _logger -> info("Starting Gaussian");
                    Gaussian gaussian;
                    output = gaussian(input, gaussian_width);
                    break;
                case gaussianSinc:
                    _logger -> info("Starting GaussianSinc");
                    GaussianSinc gaussianSinc;
                    output = gaussianSinc(input, width_normalization_gaussian, width_normalization_sinc);
                    break;
                case kernel_tophat:
                    _logger -> info("Starting Tophat Kernel");           
                    output =  make_kernel_array<TopHat>(support_tophat, input, oversampling_tophat, 0.7, false, false);
                    break;
                case kernel_triangle:
                    _logger -> info("Starting Triangle kernel");
                    output = make_kernel_array<Triangle>(support_triangle, input, oversampling_triangle, false, false, 1.5, 1.0);
                    break;
            }
        }
        catch(std::out_of_range &e) //In case an incorrect option is choosen
        {
             _logger -> error("Incorrect mode choosen");
	    return error;
        }
   
	//Prints the input matrix
	std::cout<<"INPUT: \n";
	input.print();

        //Prints the result matrix
	std::cout<<"\n\nOUPUT: \n";
	output.print();

	return ok;
}
