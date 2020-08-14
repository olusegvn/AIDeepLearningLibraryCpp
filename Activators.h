#pragma once
#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
using namespace std;
using namespace Eigen;


struct ActivatorArgs { double lowerThreshold, upperThreshold, constant, swishBeta, eSwishBeta; };
struct Activator { MatrixXd(*activate)(MatrixXd, ActivatorArgs); MatrixXd(*derivative)(MatrixXd, ActivatorArgs); };

//
//MatrixXd _linear(MatrixXd x, ActivatorArgs args);
//MatrixXd _linearDerivative(MatrixXd x, ActivatorArgs args);
//MatrixXd _sigmoid(MatrixXd x, ActivatorArgs args);
//MatrixXd _sigmoidDerivative(MatrixXd x, ActivatorArgs args);
//MatrixXd _TANH(MatrixXd x, ActivatorArgs args);
//MatrixXd _TANHDerivative(MatrixXd x, ActivatorArgs args);
//
//


double eigenExp(double x) {
	return exp(x);
}



MatrixXd _linear(MatrixXd x, ActivatorArgs args) {
	/*
	* formula : cx
	*	- unusable in backpropagation bcause derivative = constant
	*/
	return x * args.constant;
}


MatrixXd _linearDerivative(MatrixXd x, ActivatorArgs args) {
	MatrixXd m(1, 1); m << args.constant;
	return m;
}





MatrixXd _sigmoid(MatrixXd x, ActivatorArgs args) {
	/**
	* non-linear activationl functiuon on scale 0 to 1
	*
	* :param x - input matrix
	* :return - returns the activation of the matrix
	**/

	return x.unaryExpr([](double x) {return 1 / (exp(-x) + 1); });
}

MatrixXd _sigmoidDerivative(MatrixXd x, ActivatorArgs args) {
	/**
* non-linear activationl functiuon on scale 0 to 1
*
* :param x - input matrix
* :return - returns the activatior derivative of the matrix
**/

	return _sigmoid(x, ActivatorArgs{}).unaryExpr([](double x) {return x * (1 - x); });

}

MatrixXd _binaryStep(MatrixXd x, ActivatorArgs args) {
	return round(_sigmoid(x, args).array());
}


MatrixXd _binaryStepDerivative(MatrixXd x, ActivatorArgs args) {
	return round(_sigmoidDerivative(x, args).array());
}




MatrixXd _TANH(MatrixXd x, ActivatorArgs args) {
	/*
		zero - centered activation function on scale - 1 to 1
		:param x :
	:	return : tanh x which is sinhx / coshx
	*/

	return x.unaryExpr([](double x) {return ((exp(x) - exp(-x)) / 2) / ((exp(x) + exp(-x)) / 2); });

}

MatrixXd _TANHDerivative(MatrixXd x, ActivatorArgs args) {
	/*
		zero - centered activation function on scale - 1 to 1
		:param x : input
	:	return : derivativ of TANH activator derivative of the matrix
	*/
	return _TANH(x, ActivatorArgs{}).unaryExpr([](double x) {return 1 - pow(x, 2); });
}




MatrixXd _ReLU(MatrixXd x, ActivatorArgs args) {
	/**
		Rectified Linear Unit
		scale : 0 - infinity
		: param x : array of elements to be activated.
		: param threshold : value which elements below would be conerted to.
		: return : max(x, 0), ReLU derivative
		implementation of parametric ReLU comming soon

		converting the input array to absolute- x.array().abs() - may improve results
	**/

	return (args.lowerThreshold < x.array()).select(x, 0.0f);
}


MatrixXd _ReLUDerivative(MatrixXd x, ActivatorArgs args) {
	/**
		Derivative of Rectified Linear Unit
		scale : 0 to infinity
		: param x : array of elements to be activated.
		: param lowerThreshold : value which elements below would be conerted to.
		: param upperThreshold : value which elements above would be conerted to.
		: return : 0 if number is less than 0, and 1 if greater. 0.5 if number equals 0
		implementation of parametric ReLU comming soon

		converting the input array to absolute- x.array().abs() - may improve results
	**/

	return ((args.lowerThreshold < x.array()).select(x, 0.0f).unaryExpr([](double x) {return x == 0 ? 0.5 : x; }).array() < args.upperThreshold).select(x, 1.0f);
}


MatrixXd _softmax(MatrixXd x, ActivatorArgs args) {

	MatrixXd xExp = x.unaryExpr(&eigenExp);
	return xExp / xExp.sum();
}


MatrixXd _softmaxDerivative(MatrixXd x, ActivatorArgs args) {
	MatrixXd _softmax = _sigmoid(x, ActivatorArgs{});
	return _softmax.unaryExpr([](double x) {return x * (1 - x); });
}

MatrixXd _swish(MatrixXd x, ActivatorArgs args) {
	// Activation function released by Google Brain researchers
	/*
	* bounded below but unbounded above.
	* claimed to have outperformed ReLu
	* smooth and non-monotomic
	*/
	if (args.swishBeta)
	{
		return x.array() * _sigmoid(x * args.swishBeta, ActivatorArgs{}).array();
	}
	return x.array() * _sigmoid(x, ActivatorArgs{}).array();

}

MatrixXd _swishDrivative(MatrixXd x, ActivatorArgs ARGS) {
	MatrixXd fx = _swish(x, ActivatorArgs{});
	return fx.array() + (_sigmoid(x, ActivatorArgs{}).array() * fx.unaryExpr([&](double x) {return 1 - x; }).array());
}

MatrixXd _eSwish(MatrixXd x, ActivatorArgs args) {
	/*
	* claimed to have outperformed both ReLU and Swish activation functions. details at:
	*     https://arxiv.org/pdf/1801.07145.pdf

	* unbounded above and bounded below
	* smooth and non-monotomic
	* experimnetal great choice of parameters:  1 ≤ β ≤ 2.


	*/
	return args.eSwishBeta * (x * _sigmoid(x, ActivatorArgs{}));
}

MatrixXd _eSwishDerivative(MatrixXd x, ActivatorArgs args) {
	MatrixXd fx = _eSwish(x, ActivatorArgs{});
	return fx.array() + (_sigmoid(x, ActivatorArgs{}).array() * fx.unaryExpr([&](double x) {return args.eSwishBeta - x; }).array());
}




Activator linear{ _linear, _linearDerivative };
Activator sigmoid{ _sigmoid, _sigmoidDerivative };
Activator TANH{ _TANH, _TANHDerivative };
Activator ReLU{ _ReLU, _ReLUDerivative };
Activator softmax{ _softmax, _softmaxDerivative };
Activator binaryStep{ _binaryStep, _binaryStepDerivative };
Activator swish{ _swish, _swishDrivative };
Activator e_swish{ _eSwish, _eSwishDerivative };




