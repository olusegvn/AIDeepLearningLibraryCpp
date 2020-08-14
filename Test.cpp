#include "DeepNeuralNetwork.h"
#include <cstdlib>
#include <iostream>

// load csv files cpp.
// Get human readabl csv plugin

int main()
{
		//MatrixXd m(5, 2); m << 123, 546, 453, 22345, 23, 123, 546, 453, 22345, 23;
	
	TrainingParameters tp; tp.activator = swish;	tp.learningRate = 0.00001;	tp.activatorArgs = ActivatorArgs{ 0, 1 };	tp.ADAGRAD = false; tp.cost = sumOfSquares; tp.activatorArgs.swishBeta = 2.5;
	tp.model.push_back(Layer{MatrixXd:: Random(3, 1), MatrixXd:: Random(1, 1)}) ;


	MatrixXd input(5, 3); input << 0.2, 0.1, 0.0, 0.3, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.1, 0.0, 0.2, 0.1, 0.1;
	MatrixXd labels(5, 1); labels << 0.1, 0.2, 0.0, 0.1, 0.2;
	tp.input = input; tp.labels = labels;
	
	AINeuralNetwork ai = AINeuralNetwork(tp, "simple", false);
	auto start = std::chrono::high_resolution_clock::now();
	ai.train(gradientDescent, 100, 10);
	ai.cache("model.dat");
	AINeuralNetwork pickai = AINeuralNetwork(tp, "simple", false);
	
	pickai.pick("model.dat");
	cout << pickai.predict(input);
	auto finish = std::chrono::high_resolution_clock::now();
	duration<double> elapsed = finish - start;
	std::cout << "\nElapsed time: " << elapsed.count() << " s\n";
	cout << "passed";

}



