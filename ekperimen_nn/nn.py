# ==========================================
# 1. PERHITUNGAN MATEMATIKA TANPA MODUL 'MATH'
# ==========================================

def exp(x, terms=20):
    """
    Menghitung e^x menggunakan Taylor Series Expansion:
    e^x = 1 + x + x^2/2! + x^3/3! + ...
    """
    # Untuk penanganan nilai x negatif yang besar agar stabil
    if x < -20:
        return 0.0
    if x < 0:
        return 1.0 / exp(-x, terms)
    
    result = 1.0
    term = 1.0
    for i in range(1, terms):
        term *= x / i
        result += term
    return result

def sigmoid(x):
    """Fungsi Aktivasi Sigmoid: 1 / (1 + e^-x)"""
    return 1.0 / (1.0 + exp(-x))

def sigmoid_derivative(x):
    """Turunan Sigmoid dari nilai keluaran aktivasi"""
    return x * (1.0 - x)

# ==========================================
# 2. RANDOM GENERATOR TANPA MODUL 'RANDOM'
# ==========================================

class SimpleRandom:
    """Linear Congruential Generator (LCG) untuk angka acak semu"""
    def __init__(self, seed=42):
        self.state = seed

    def random(self):
        # Konstanta standar LCG
        self.state = (1664525 * self.state + 1013904223) % 4294967296
        return self.state / 4294967296.0

    def uniform(self, a, b):
        return a + (b - a) * self.random()

# Inisialisasi generator acak
rng = SimpleRandom(seed=12345)

# ==========================================
# 3. STRUKTUR NEURAL NETWORK
# ==========================================

def dot_product(vector_a, vector_b):
    return sum(a * b for a, b in zip(vector_a, vector_b))

def random_matrix(rows, cols):
    return [[rng.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]

class PureNeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_input_hidden = random_matrix(self.input_nodes, self.hidden_nodes)
        self.weights_hidden_output = random_matrix(self.hidden_nodes, self.output_nodes)
        
        self.bias_hidden = [rng.uniform(-1, 1) for _ in range(self.hidden_nodes)]
        self.bias_output = [rng.uniform(-1, 1) for _ in range(self.output_nodes)]

    def feedforward(self, input_vector):
        # Input ke Hidden Layer
        hidden_inputs = []
        for j in range(self.hidden_nodes):
            column = [self.weights_input_hidden[i][j] for i in range(self.input_nodes)]
            val = dot_product(input_vector, column) + self.bias_hidden[j]
            hidden_inputs.append(val)
        
        hidden_outputs = [sigmoid(x) for x in hidden_inputs]

        # Hidden ke Output Layer
        final_inputs = []
        for k in range(self.output_nodes):
            column = [self.weights_hidden_output[j][k] for j in range(self.hidden_nodes)]
            val = dot_product(hidden_outputs, column) + self.bias_output[k]
            final_inputs.append(val)
            
        final_outputs = [sigmoid(x) for x in final_inputs]

        return hidden_outputs, final_outputs

    def train(self, inputs_list, targets_list, learning_rate, epochs):
        for epoch in range(epochs):
            total_error = 0
            
            for input_vector, target_vector in zip(inputs_list, targets_list):
                # Feedforward
                hidden_outputs, final_outputs = self.feedforward(input_vector)

                # Error Output
                output_errors = [target - actual for target, actual in zip(target_vector, final_outputs)]
                total_error += sum(0.5 * (err ** 2) for err in output_errors)

                # Gradien Output
                output_gradients = [
                    err * sigmoid_derivative(out) * learning_rate 
                    for err, out in zip(output_errors, final_outputs)
                ]

                # Error Hidden
                hidden_errors = []
                for j in range(self.hidden_nodes):
                    error = sum(
                        output_errors[k] * self.weights_hidden_output[j][k] 
                        for k in range(self.output_nodes)
                    )
                    hidden_errors.append(error)

                # Gradien Hidden
                hidden_gradients = [
                    err * sigmoid_derivative(out) * learning_rate 
                    for err, out in zip(hidden_errors, hidden_outputs)
                ]

                # Update Bobot Hidden -> Output
                for j in range(self.hidden_nodes):
                    for k in range(self.output_nodes):
                        self.weights_hidden_output[j][k] += output_gradients[k] * hidden_outputs[j]
                
                for k in range(self.output_nodes):
                    self.bias_output[k] += output_gradients[k]

                # Update Bobot Input -> Hidden
                for i in range(self.input_nodes):
                    for j in range(self.hidden_nodes):
                        self.weights_input_hidden[i][j] += hidden_gradients[j] * input_vector[i]
                
                for j in range(self.hidden_nodes):
                    self.bias_hidden[j] += hidden_gradients[j]

            if (epoch + 1) % 2000 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_error:.5f}")

    def predict(self, input_vector):
        _, final_outputs = self.feedforward(input_vector)
        return final_outputs

# ==========================================
# 4. UJI COBA PADA GERBANG XOR
# ==========================================

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]

nn = PureNeuralNetwork(input_nodes=2, hidden_nodes=4, output_nodes=1)

print("--- Pelatihan Neural Network Murni ---")
nn.train(X, Y, learning_rate=0.5, epochs=10000)

print("\n--- Hasil Evaluasi Prediksi ---")
for inputs in X:
    prediction = nn.predict(inputs)
    val = prediction[0]
    rounded = 1 if val >= 0.5 else 0
    print(f"Input: {inputs} -> Output: {val:.4f} (Klasifikasi: {rounded})")
