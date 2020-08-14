#pragma once


#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <chrono> 
#include <cmath>
using namespace std::chrono;
using namespace std;
using namespace Eigen;


struct Cost { MatrixXd(*error)(MatrixXd, MatrixXd); MatrixXd(*derivative)(MatrixXd, MatrixXd); };

MatrixXd _difference(MatrixXd z, MatrixXd labels) {
	/*
	 - the difference cost function is both the cost and the derivative.
	*/

	return z - labels;
}



MatrixXd _sumOfSquares(MatrixXd z, MatrixXd labels) {
	return (z - labels).unaryExpr([](double x) {return pow(x, 2); });
}


MatrixXd _sumOfSquaresDerivative(MatrixXd z, MatrixXd labels) {
	return  2 * (z - labels);
}



void _crossEntropy(MatrixXd z, MatrixXd label) {
	// gray area
}

MatrixXd _absoluteError(MatrixXd z, MatrixXd label) {
	/*
	 - the absolute error cost function is both the cost and the derivative.
	*/
	return (z - label).cwiseAbs();
}




Cost difference{ _difference, _difference };
Cost sumOfSquares{ _sumOfSquares, _sumOfSquaresDerivative };
Cost crossEntropy{ _absoluteError, _absoluteError };








