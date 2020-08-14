#pragma once







// conform if training with bes6t model i sbetter than training with model


//
//
//class AINeuralNetwork {
//public:
//	string type;
//	bool fine;
//	TrainingParameters trainingParameters;
//	vector<double> errorHistory;
//	bool highLearningRate;
//
//
//	AINeuralNetwork(TrainingParameters _trainingParameters, string _type="perceptron", bool _fine=false) 
//	{
//		fine = _fine;
//		type = _type;
//		trainingParameters = _trainingParameters;
//
//	}
//
//	// TODO: 
//
//	// check this
//	void cache(string filename="model.dat") 
//	{
//		ofstream fout(filename);
//		fout.write(reinterpret_cast<char*>(&trainingParameters), sizeof(trainingParameters));
//	}
//
//	// check this
//	void pick(string filename= "model.dat")
//	{
//		ifstream fin(filename);
//		fin.read(reinterpret_cast<char*>(&trainingParameters), sizeof(trainingParameters));
//
//	}
//
//
//	void train( OptimizedModel(*optimizer)(TrainingParameters),int epochs=1000, int showStep=10, string interruptSave="interruptSavedModel.dat", int batchRows=0, bool showErrorHistory=true) {
//		batchRows = batchRows ? trainingParameters.input.rows() : batchRows;
//		string perceptronNames[] = {"simple", "perceptron"};
//		if (find(perceptronNames, &perceptronNames[3], type) != &perceptronNames[2])
//			trainingParameters.model = { Layer{MatrixXd::Random(trainingParameters.perceptronSize, 1), MatrixXd::Random(1)} };
//		double bestError = numeric_limits<double>::infinity();
//
//		// TODO: fine
//
//		for (int _ = 1; _ <= epochs; _++) 
//		{
//			// time execution for ETA.
//			steady_clock::time_point start = chrono::high_resolution_clock::now();
//			VectorXd totalError; int batchStart = 0, batchEnd = batchRows;
//			// Batching.
//			for (size_t __ = 0; __ < trainingParameters.input.cols()/batchRows; __++)
//			{
//				OptimizedModel optimizedModel = optimizer(trainingParameters);
//				// optimize line to avoid reinitialization.
//				trainingParameters.model = optimizedModel.model;
//				totalError[__] = optimizedModel.error.sum();
//				batchStart, batchEnd += batchRows;
//
//			}
//			double error =  totalError.cwiseAbs().mean();
//			errorHistory[_] = error;
//				
//			if (error < bestError) {
//				bestError = error;
//				trainingParameters.bestModel = trainingParameters.model;
//			}
//
//
//			if (abs(error) > abs(bestError))
//				highLearningRate = true;
//			if ((_) % showStep == 0 || _ == 1)
//			{
//				cout << "Epoch: " << _ << "\t|   Error:  " << error << '\t';
//				steady_clock::time_point end = chrono::high_resolution_clock::now();
//				double seconds = chrono::duration_cast<chrono::seconds>(end - start).count();
//				showETA(seconds);
//				if(showErrorHistory)
//				{
//					cache(interruptSave);
//
//				}
//			}
//			if(highLearningRate)
//			{
//				cout << "\033[31mLearning rate too high, reduce for better performance\033[0m\n";
//				cout << "\033[34mBest error :  "<< bestError<<"\033[0m\n";
//			}
//		
//			
//		}
//	}
//
//	// predict
//
//};
//
//
//
//
//
//
//int main() {	
//	//MatrixXd m(5, 2); m << 123, 546, 453, 22345, 23, 123, 546, 453, 22345, 23;
//	//auto start = std::chrono::high_resolution_clock::now();
//	
//	TrainingParameters tp; tp.activator = ReLU;	tp.learningRate = 0.00001;	tp.activatorArgs = ActivatorArgs{0, 1};	tp.ADAGRAD = false;
//	AINeuralNetwork ai = AINeuralNetwork(tp, "simple", false);
//	ai.train(gradientDescent, 1000, 100);
//
//	//auto finish = std::chrono::high_resolution_clock::now();
//	//duration<double> elapsed = finish - start;
//	//std::cout << "\nElapsed time: " << elapsed.count() << " s\n";
//}