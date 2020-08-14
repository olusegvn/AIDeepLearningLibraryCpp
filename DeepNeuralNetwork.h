#pragma once

#include "Activators.h"
#include "Costs.h"
#include "Optimizers.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <variant>
using namespace std::chrono;
using namespace std;

void showETA(int n)
{
	int day = n / (24 * 3600);

	n = n % (24 * 3600);
	int hour = n / 3600;

	n %= 3600;
	int minutes = n / 60;

	n %= 60;
	int seconds = n;

	cout << "ETA : " << day << " " << "days " << hour
		<< " " << "hours " << minutes << " "
		<< "minutes " << seconds << " "
		<< "seconds " << endl;
}


class AINeuralNetwork {
public:
	string type;
	bool fine;
	TrainingParameters trainingParameters;
	vector<double> errorHistory;
	bool highLearningRate;

	AINeuralNetwork(TrainingParameters _trainingParameters, string _type = "perceptron", bool _fine = false)
	{
		fine = _fine;
		type = _type;
		trainingParameters = _trainingParameters;

	}

	void cache(string filename="model.dat") 
	{
		ofstream fout(filename);
		fout.write(reinterpret_cast<char*>(&trainingParameters), sizeof(trainingParameters));
	}

	// check this
	void pick(string filename= "model.dat")
	{
		ifstream fin(filename);
		fin.read(reinterpret_cast<char*>(&trainingParameters), sizeof(trainingParameters));

	}


	void train(OptimizedModel(*optimizer)(TrainingParameters), int epochs = 1000, int showStep = 10, string interruptSave = "interruptSavedModel.dat", int batchRows = 0, bool showErrorHistory = true)
	{
		
		batchRows = batchRows<=0 ? trainingParameters.input.rows() : batchRows;
		
		if (type == "perceptron")
			trainingParameters.model = { Layer{MatrixXd::Random(trainingParameters.perceptronSize, 1), MatrixXd::Random(1, 1)} };
		double bestError = numeric_limits<double>::infinity();
		
		// TODO: fine

		for (int _ = 1; _ <= epochs; _++)
		{
			// time execution for ETA.
			
			steady_clock::time_point start = chrono::high_resolution_clock::now();
			int batch = trainingParameters.input.rows() / batchRows;
			VectorXd totalError(batch); int batchStart = 0, batchEnd = batchRows;
			// Batching.
			for (size_t __ = 0; __ < trainingParameters.input.rows()/batchRows; __++)
			{
				OptimizedModel optimizedModel = optimizer(trainingParameters);

				// optimize line to avoid reinitialization.
				trainingParameters.model = optimizedModel.model;
				totalError[__] =  optimizedModel.error.sum();

				batchStart, batchEnd += batchRows;
			
			}

			double error =  totalError.cwiseAbs().mean();
			errorHistory.push_back(error);

			if (error < bestError) {
				bestError = error;
				trainingParameters.bestModel = trainingParameters.model;
			}
			

			if (abs(error) > abs(bestError))
				highLearningRate = true;
			if ((_) % showStep == 0 || _ == 1)
			{
				cout << "Epoch: " << _ << "\t|   Error:  " << error << '\t';
				steady_clock::time_point end = chrono::high_resolution_clock::now();
				double seconds = chrono::duration_cast<chrono::seconds>(end - start).count();
				showETA(seconds);
				if(showErrorHistory)
				{
					cache(interruptSave);
			
				}
			}
										
		}
		if (highLearningRate)
		{
			cout << "\033[31mLearning rate too high, reduce for better performance\033[0m\n";
			cout << "\033[34mBest error :  " << bestError << "\033[0m\n";
		}
	}

	MatrixXd predict(MatrixXd input)
	{
		TrainingParameters tp = trainingParameters;
		tp.input = input;
		return feedForward(trainingParameters, true);
	}

};