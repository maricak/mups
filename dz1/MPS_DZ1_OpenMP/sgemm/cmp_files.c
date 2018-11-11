#include <fstream>
#include <cmath>
#include <iostream>


bool compareFiles(const char* f1, const char* f2) {
  
  const double ACCURACY = 0.0100000000001;

  std::fstream file1(f1, std::fstream::in);
  if ( !file1.good() ) {
    return false;
  }

	std::fstream file2(f2, std::fstream::in);
  if ( !file2.good() ) {
    return false;
  }

	double data1, data2;
	while (file1.good() && file2.good()) {
		file1 >> data1;
		file2 >> data2;

		if (fabs(data1 - data2) >= ACCURACY) {
			return false;
		}
	}

	if (file1.eof() && file2.eof()) {
			return true;
	}

  return false;
}


int main(int argc, char** argv) {

	if (argc != 3) {
		std::cerr << "Invalid number of arguments." << std::endl;
	}


	if (compareFiles(argv[1], argv[2])) {
		std::cerr << "Test PASSED" << std::endl;
	}
	else {
		std::cerr << "Test FAILED" << std::endl;
	}
	
	return 0;
}
