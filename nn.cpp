#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

double sigmoid(double x) {
return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
return x * (1.0 - x);
}

class Layer {
public:
int num_neurons;
int num_inputs_per_neuron;
vector<vector<double>> weights;
vector<double> biases;
vector<double> outputs;
vector<double> deltas;

Layer(int n, int inputs) : num_neurons(n), num_inputs_per_neuron(inputs) {  
    weights.resize(n, vector<double>(inputs));  
    biases.resize(n);  
    outputs.resize(n);  
    deltas.resize(n);  

    for (int i = 0; i < n; i++) {  
        biases[i] = ((double)rand() / RAND_MAX) * 2 - 1;  
        for (int j = 0; j < inputs; j++) {  
            weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;  
        }  
    }  
}  

void forward(const vector<double>& inputs) {  
    for (int i = 0; i < num_neurons; i++) {  
        double activation = biases[i];  
        for (int j = 0; j < num_inputs_per_neuron; j++) {  
            activation += inputs[j] * weights[i][j];  
        }  
        outputs[i] = sigmoid(activation);  
    }  
}

};

class NeuralNetwork {
public:
Layer hidden1;
Layer hidden2;
Layer output_layer;

// Arsitektur: 1 Input -> 128 Hidden -> 128 Hidden -> 1 Output  
NeuralNetwork() : hidden1(128, 1), hidden2(128, 128), output_layer(1, 128) {}  

double predict(double input) {  
    vector<double> in = {input};  
    hidden1.forward(in);  
    hidden2.forward(hidden1.outputs);  
    output_layer.forward(hidden2.outputs);  
    return output_layer.outputs[0];  
}  

void train(double input, double target, double lr) {  
    // 1. Forward Pass  
    double prediction = predict(input);  

    // 2. Backpropagation  
    // Output Layer Delta  
    output_layer.deltas[0] = (target - prediction) * sigmoid_derivative(prediction);  

    // Hidden 2 Delta  
    for (int i = 0; i < hidden2.num_neurons; i++) {  
        double error = output_layer.deltas[0] * output_layer.weights[0][i];  
        hidden2.deltas[i] = error * sigmoid_derivative(hidden2.outputs[i]);  
    }  

    // Hidden 1 Delta  
    for (int i = 0; i < hidden1.num_neurons; i++) {  
        double error = 0;  
        for (int j = 0; j < hidden2.num_neurons; j++) {  
            error += hidden2.deltas[j] * hidden2.weights[j][i];  
        }  
        hidden1.deltas[i] = error * sigmoid_derivative(hidden1.outputs[i]);  
    }  

    // 3. Update Weights & Biases  
    auto update = [&](Layer& l, const vector<double>& prev_outputs) {  
        for (int i = 0; i < l.num_neurons; i++) {  
            for (int j = 0; j < l.num_inputs_per_neuron; j++) {  
                l.weights[i][j] += lr * l.deltas[i] * prev_outputs[j];  
            }  
            l.biases[i] += lr * l.deltas[i];  
        }  
    };  

    update(output_layer, hidden2.outputs);  
    update(hidden2, hidden1.outputs);  
    update(hidden1, {input});  
}

};

int main() {
srand(time(NULL));
NeuralNetwork nn;
double learning_rate = 0.5;

cout << "Training 128-neuron MLP di HP..." << endl;  

// Latihan: Jika x < 0.5 maka 0, jika x >= 0.5 maka 1  
for (int i = 0; i < 20000; i++) {  
    double x = (double)rand() / RAND_MAX;  
    double y = (x >= 0.5) ? 1.0 : 0.0;  
    nn.train(x, y, learning_rate);  
}  

cout << "Selesai. Hasil test:" << endl;  
cout << "Input 0.1 -> Prediksi: " << nn.predict(0.1) << " (Target: 0)" << endl;  
cout << "Input 0.9 -> Prediksi: " << nn.predict(0.9) << " (Target: 1)" << endl;  

return 0;

}
