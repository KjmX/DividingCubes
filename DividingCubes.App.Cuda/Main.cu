#include "CudaDividingCubesApp.cuh"

#include "Parser\argvparser.h"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace Impacts::Apps;
using namespace CommandLineProcessing;


string getOptionValue(ArgvParser const & parser, string option)
{
	if (parser.foundOption(option))
		return parser.optionValue(option);

	return "";
}


int main(int argc, char ** argv)
{
	/*
	string help = "\
	-ww [[Window Width]]\n\
	-wh [[Window Height]]\n\
	-d [[Dataset Path]]\n\
	-dx [[Dimension x-axis]]\n\
	-dy [[Dimension y-axis]]\n\
	-dz [[Dimension z-axis]]\n\
	-csx [[Cude size X]]\n\
	-csy [[Cude size Y]]\n\
	-csz [[Cude size Z]]\n\
	-iso [[Isovalue]]\n\
	-sd [[Sub distance]]\n\
	";

	ArgvParser parser;

	parser.setIntroductoryDescription("This is a test");

	parser.addErrorCode(0, "Success");
	parser.addErrorCode(1, "Error");

	//parser.setHelpOption("h", "help", help);


	parser.defineOption("ww", "", ArgvParser::OptionRequiresValue);
	parser.defineOption("wh", "768", ArgvParser::OptionRequiresValue);
	parser.defineOption("d", "", ArgvParser::OptionRequiresValue);
	parser.defineOption("dx", "", ArgvParser::OptionRequiresValue);
	parser.defineOption("dy", "", ArgvParser::OptionRequiresValue);
	parser.defineOption("dz", "", ArgvParser::OptionRequiresValue);
	parser.defineOption("csx", "", ArgvParser::OptionRequiresValue);
	parser.defineOption("csy", "", ArgvParser::OptionRequiresValue);
	parser.defineOption("csz", "", ArgvParser::OptionRequiresValue);
	parser.defineOption("iso", "", ArgvParser::OptionRequiresValue);
	parser.defineOption("sd", "", ArgvParser::OptionRequiresValue);

	auto result = parser.parse(argc, argv);

	if (result != ArgvParser::NoParserError)
	{
		cout << parser.parseErrorDescription(result);
		return 0;
	}

	auto width = getOptionValue(parser, "width");
	auto height = getOptionValue(parser, "wh");
	auto datasetPath = getOptionValue(parser, "d");
	auto dimX = getOptionValue(parser, "dx");
	auto dimY = getOptionValue(parser, "dy");
	auto dimZ = getOptionValue(parser, "dz");
	auto cubeSizeX = getOptionValue(parser, "csx");
	auto cubeSizeY = getOptionValue(parser, "csy");
	auto cubeSizeZ = getOptionValue(parser, "csz");
	auto isoValue = getOptionValue(parser, "iso");
	auto subDistance = getOptionValue(parser, "sd");*/

	try
	{
		CudaDividingCubesApp app(1024, 768, "", glm::tvec3<int>(), glm::vec3(), 40.f, 1.f);
		app.Run();
	}
	catch (runtime_error & e)
	{
		cout << e.what() << endl;
	}
	

	return 0;
}