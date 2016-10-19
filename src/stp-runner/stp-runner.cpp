/**
*	@file stp-runner.cpp
*	Main file of the stp runner
*	Contains the main function. Creates and configures the TCLAP and Spdlog interface. Calls test functions.
*/

#include "stp-runner.h"

//Logger
shared_ptr <logger> _logger;

//Selection flag
ValueArg<string> _modeFlag("m", "mode", "Chooses witch function to use\ntophat for tophat\nssinc for sinc\ngaussian for gaussian\ntriangle for triangle\ngaussian-sinc for gaussian-sinc\ntophat-kernel to create a tophat kernel\ntriangle-kernel to create a triangle kernel", true, "tophat", "string");

//Filepath flag
ValueArg<string> _fileArg("f","filepath","Input filepath",true,"../mock_uvw_vis.npz","string");

//Key flag
ValueArg<string> _keyArg("k","key","Key for npz file",false, "false", "string");

//Print to file flag
SwitchArg _printArg("p", "print", "Print to file", false);

//Command line parser
CmdLine _cmd("Parser for the stp runner", ' ', "0.2");

/**
*	@brief Auxiliary function to evaluate a string
*	
*	Receives a string and recursively evaluates it to an integer
*	
*	@param[in] str String to be evaluated
*	@param [in,out] h Position of string that is being evaluated
*
*	@return An integer that corresponds to the string
*/
constexpr unsigned int str2int(const char* str, int h = 0)
{
    return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h];
}

/**
*	@brief Creates the switch flags to be used by the parser
*
*	Creates a list of switch arguments and then xor adds to the command line parser, meaning that only
*	one flag and one only is required to run the program_invocation_name
*
*/
void createFlags() throw(ArgException)
{
	//Adds the mode flag to the parse. It chooses which function I wish to call from the libstp
	_logger -> debug("Adding the mode flag to the command line parser");
	_cmd.add(_modeFlag);

	//Adds the value flag to the parser list. This is flag is required
	_logger -> debug("Adding the filepath flag to the command line parser");
	_cmd.add(_fileArg);

	//Adds the key flag to the parser list. At least one key is required
	_logger -> debug("Adding the key flag to the command line parser");
	_cmd.add(_keyArg);

	//Adds the print to file flag to the parser list. If enabled it prints the output to a file
	_logger -> debug("Adding the print to file flag to the command line parser");
	_cmd.add(_printArg);
}

/**
* 	@brief Logger initialization function
*
*	Creates and initializes the logger to be used throughout the program
*/
void initLogger() throw(ArgException)
{
	//Creates two spdlog sinks
	//One sink for the stdout and another for a file
	vector <sink_ptr> sinks;
	sinks.push_back(make_shared <sinks::stdout_sink_st> ());
	sinks.push_back(make_shared <sinks::simple_file_sink_st> ("logfile.txt"));
	_logger = make_shared <spdlog::logger> ("logger", begin(sinks), end(sinks));
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
	try
	{
		initLogger();
	}
	catch(ArgException &e)
	{
		_logger -> critical("Cannot create share logger sink. Aborting");
		_logger -> error("error: " + e.error() + "for: " + e.argId());
		return error;
	}
	_logger -> info("Program start");

	//Adds the flags to the parser
	try
	{
		createFlags();
	}
	catch(ArgException &e)
	{
		_logger -> critical("Cannot properly configure parsing flags. Aborting");
		_logger -> error("error: " + e.error() + "for: " + e.argId());
		return error;
	}

	//Parses the arguments from the command console
	_logger -> info("Parsing arguments from console");
	try
	{
		_cmd.parse(argc, argv);
	}
	catch(ArgException &e)
	{
		_logger -> critical("Error reading arguments from command line. Aborting");
		_logger -> error("error: " + e.error() + "for: " + e.argId());
	}

	//Creates the input armadillo matrix for the convolution functions
	mat input;

	//If a key is present loads an npz, otherwise loads an npy

	if (_keyArg.getValue().compare("false") != 0)
	{
	    NpyArray npz_data = npz_load(_fileArg.getValue(), _keyArg.getValue());
	    input = loadNpyArray(&npz_data);
	}
	else
	{
	    NpyArray npy_data = npy_load(_fileArg.getValue());
	    input = loadNpyArray(&npy_data);
	}

	//Creates the output armadillo matrix
	mat output;
	
	//Chooses which function to run
	switch (str2int(_modeFlag.getValue().c_str()))
	{
	case str2int("tophat"):
	    _logger -> info("Starting Tophat");
	    output = make_conv_func_tophat(3.0, input);
	    break;
	case str2int("triangle"):
	    _logger -> info("Starting Triangle");
		output = make_conv_func_triangle(3.0, input, 1.0);
	    break;
	case str2int("sinc"):
	    _logger -> info("Starting Sinc");
		output = make_conv_func_sinc(input);
	    break;
	case str2int("gaussian"):
	    _logger -> info("Starting Gaussian");
		output = make_conv_func_gaussian(input, 1.0);
	    break;
	case str2int("gaussian-sinc"):
	    _logger -> info("Starting Gaussian-sinc");
		output = make_conv_func_gaussian_sinc(input, 2.52, 1.55);
	    break;
	case str2int("tophat-kernel"):
	    _logger -> info("Starting Tophat Kernel");
		output = make_top_hat_kernel_array(1, input, 3.0, 0.7, false, false);
	    break;
	case str2int("triangle-kernel"):
	    _logger -> info("Starting Triangle kernel");
		output = make_triangle_kernel_array(2, input, 1.0, 1.5, false, false);
	    break;
	default:
	    _logger -> error("Incorrect mode choosen");
	    return error;
	}

	//Prints the input matrix
	cout<<"INPUT: \n";
	input.print();

	//If print to file flag is on, redirect print to file
	if(_printArg.getValue())
	{
		ofstream out("output.txt");
		streambuf *coutbuf = std::cout.rdbuf();
		cout.rdbuf(out.rdbuf());
		output.print();
		cout.rdbuf(coutbuf);
	}
	cout<<"\n\nOUPUT: \n";
	output.print();

	return ok;
}
