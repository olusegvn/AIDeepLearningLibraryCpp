#pragma once
#include "Costs.h"
#include "Activators.h"
#include <vector>
#include <chrono> 
#include <Eigen/Dense>
#include <iostream>
using namespace std::chrono;
using namespace std;
using namespace Eigen;



// attemt writing a new optimizer, it may be faster than debugging.

// Algorithm for numpy like matrix multiplication



struct Layer { MatrixXd weights; MatrixXd bias; };
struct OptimizedModel { MatrixXd error;  vector<Layer> model; };
struct TrainingParameters { vector<Layer> model, bestModel; MatrixXd input, labels; int perceptronSize; Cost cost; double learningRate, e; bool ADAGRAD; Activator activator; ActivatorArgs activatorArgs; };

MatrixXd feedForward(TrainingParameters trainingParameters, bool bestModel = false) {
	for (Layer layer : bestModel ? trainingParameters.bestModel : trainingParameters.model)
	{
		MatrixXd dotProduct; 
		dotProduct = trainingParameters.input * layer.weights;
		//trainingParameters.input = trainingParameters.activator.activate(dotProduct + layer.bias, trainingParameters.activatorArgs);

		// confirm math

		trainingParameters.input = trainingParameters.activator.activate(dotProduct.unaryExpr([&](double x) {return x + layer.bias(0); }), trainingParameters.activatorArgs);

	}
	

	return trainingParameters.input;
}




OptimizedModel gradientDescent(TrainingParameters trainingParameters) {
	MatrixXd z = feedForward(trainingParameters);

	//cout << trainingParameters.model[0].weights;
	for (int _ = 0; _ < trainingParameters.model.size(); _++) {
		MatrixXd gradient, gradientBias;

		MatrixXd dcost_dz = trainingParameters.cost.derivative(z, trainingParameters.labels);

		MatrixXd dz_dy = trainingParameters.activator.derivative(z, trainingParameters.activatorArgs);
		MatrixXd dy_dw = trainingParameters.input.transpose();

		// implement numpy-like matrix nultiplication 

		MatrixXd d1 = dz_dy.array() * dcost_dz.array();


		// confirm that math substitutes dot product 
		MatrixXd dcost_dw = dy_dw * d1;


		if (trainingParameters.ADAGRAD) {

		}
		else {

			MatrixXd _dcost_dw = dcost_dw.array() * trainingParameters.learningRate;
			trainingParameters.model[_].weights -= dcost_dw;
			//cout << layer.weights;

			//layer.weights = layer.weights.unaryExpr([&](double x) {return x - _dcost_dw; });
			for (size_t i = 0, nRows = d1.rows(), nCols = d1.cols(); i < nRows; i++)
				for (size_t j = 0; j < nCols; ++j)
				{

					double _value = d1(i, j) * trainingParameters.learningRate;
					trainingParameters.model[_].bias = trainingParameters.model[_].bias.unaryExpr([&](double x) {return x - _value; });
				}
		}
		trainingParameters.input = z;

	}
	//cout << "\n\nnew weights : "<< trainingParameters.model[0].weights << "\n\n";
	return OptimizedModel{ trainingParameters.cost.error(z, trainingParameters.labels), trainingParameters.model };
}





