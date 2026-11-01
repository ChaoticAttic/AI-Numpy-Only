import numpy as np
import json
import sys
import time
from data_gen import gen_mixed_points

with open("architecture.json", "r") as f:
    nn_architecture = json.load(f)

def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
        
    return params_values

def sigmoid(Z):
    result = np.empty_like(Z, dtype=float)
    positive = Z >= 0
    result[positive] = 1 / (1 + np.exp(-Z[positive]))
    exp_z = np.exp(Z[~positive])
    result[~positive] = exp_z / (1 + exp_z)
    return result


def softmax(Z):
    shifted_Z = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(shifted_Z)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def softmax_backward(dA, Z):
    probabilities = softmax(Z)
    weighted_gradient = np.sum(dA * probabilities, axis=0, keepdims=True)
    return probabilities * (dA - weighted_gradient)


def relu(Z):
    return np.maximum(0,Z)
def sin(Z):
    return np.sin(Z)


def sin_backward(dA, Z):
    return dA * np.cos(Z)


def rho1(Z):
    result = np.empty_like(Z, dtype=float)
    non_positive = Z <= 0
    result[non_positive] = 1 - Z[non_positive]
    positive = Z > 0
    result[positive] = (1 + Z[positive]) / (Z[positive] ** 2 + 9)
    return result

def sin(Z):
    return np.sin(Z)

def rho1_backward(dA, Z):
    derivative = np.empty_like(Z, dtype=float)
    non_positive = Z <= 0
    derivative[non_positive] = -1
    positive = Z > 0
    denominator = Z[positive] ** 2 + 9
    derivative[positive] = (
        9 - 2 * Z[positive] - Z[positive] ** 2
    ) / denominator**2
    return dA * derivative


def gelu(Z):
    coefficient = np.sqrt(2 / np.pi)
    inner = coefficient * (Z + 0.044715 * Z**3)
    return 0.5 * Z * (1 + np.tanh(inner))


def gelu_backward(dA, Z):
    coefficient = np.sqrt(2 / np.pi)
    inner = coefficient * (Z + 0.044715 * Z**3)
    inner_derivative = coefficient * (1 + 3 * 0.044715 * Z**2)
    tanh_inner = np.tanh(inner)
    derivative = (
        0.5 * (1 + tanh_inner)
        + 0.5 * Z * (1 - tanh_inner**2) * inner_derivative
    )
    return dA * derivative


def mish(Z):
    softplus_Z = np.logaddexp(0, Z)
    return Z * np.tanh(softplus_Z)


def mish_backward(dA, Z):
    softplus_Z = np.logaddexp(0, Z)
    tanh_softplus = np.tanh(softplus_Z)
    sigmoid_Z = sigmoid(Z)
    derivative = (
        tanh_softplus
        + Z * sigmoid_Z * (1 - tanh_softplus**2)
    )
    return dA * derivative


def softsign(Z):
    return Z / (1 + np.abs(Z))


def softsign_backward(dA, Z):
    return dA / (1 + np.abs(Z))**2

def sin_backwards(dA, Z):
    return dA * np.cos(Z)

def swish_gaussian(Z):
    sigmoid_Z = sigmoid(Z)
    return Z * sigmoid_Z + np.cos(Z) * np.exp(-0.1 * Z**2)


def swish_gaussian_backward(dA, Z):
    sigmoid_Z = sigmoid(Z)
    derivative = (
        sigmoid_Z
        + Z * sigmoid_Z * (1 - sigmoid_Z)
        + np.exp(-0.1 * Z**2) * (-np.sin(Z) - 0.2 * Z * np.cos(Z))
    )
    return dA * derivative


def softplus_wave(Z):
    phase = 3 * Z + np.sin(Z)
    return np.logaddexp(0, Z) + np.sin(phase) / (1 + Z**2)


def softplus_wave_backward(dA, Z):
    phase = 3 * Z + np.sin(Z)
    denominator = 1 + Z**2
    derivative = sigmoid(Z) + (
        np.cos(phase) * (3 + np.cos(Z)) * denominator
        - 2 * Z * np.sin(phase)
    ) / denominator**2
    return dA * derivative


def peir(Z):
    numerator = np.cos(np.sqrt(3) * Z) * np.sin(np.pi * Z)
    return np.arctan(Z) + numerator / (2 + np.abs(Z))


def peir_backward(dA, Z):
    sqrt_3_Z = np.sqrt(3) * Z
    pi_Z = np.pi * Z
    numerator = np.cos(sqrt_3_Z) * np.sin(pi_Z)
    numerator_derivative = (
        -np.sqrt(3) * np.sin(sqrt_3_Z) * np.sin(pi_Z)
        + np.pi * np.cos(sqrt_3_Z) * np.cos(pi_Z)
    )
    denominator = 2 + np.abs(Z)
    derivative = (
        1 / (1 + Z**2)
        + numerator_derivative / denominator
        - numerator * np.sign(Z) / denominator**2
    )
    return dA * derivative


ACTIVATION_EXP_LIMIT = 20.0
ACTIVATION_TAN_LIMIT = 1e6


def _safe_tan(Z):
    return np.nan_to_num(
        np.tan(Z),
        nan=0.0,
        posinf=ACTIVATION_TAN_LIMIT,
        neginf=-ACTIVATION_TAN_LIMIT,
    )


def arctan_wave(Z):
    wave = (
        3 * np.sin(Z)
        + 2 * np.sin(np.sqrt(2) * Z)
        + np.sin(np.pi * Z)
    )
    return np.arctan(Z) + np.arctan(wave)


def arctan_wave_backward(dA, Z):
    wave = (
        3 * np.sin(Z)
        + 2 * np.sin(np.sqrt(2) * Z)
        + np.sin(np.pi * Z)
    )
    wave_derivative = (
        3 * np.cos(Z)
        + 2 * np.sqrt(2) * np.cos(np.sqrt(2) * Z)
        + np.pi * np.cos(np.pi * Z)
    )
    derivative = 1 / (1 + Z**2) + wave_derivative / (1 + wave**2)
    return dA * derivative


def wrapped_exp_cos(Z):
    exponent = np.clip(Z * np.sin(Z), -ACTIVATION_EXP_LIMIT,
                       ACTIVATION_EXP_LIMIT)
    first_term = np.arctan(np.tan(np.exp(exponent)))
    second_argument = _safe_tan(Z / (1 + 0.1 * Z**2))
    return first_term * np.cos(second_argument)


def wrapped_exp_cos_backward(dA, Z):
    exponent = np.clip(Z * np.sin(Z), -ACTIVATION_EXP_LIMIT,
                       ACTIVATION_EXP_LIMIT)
    exp_value = np.exp(exponent)
    exponent_derivative = np.sin(Z) + Z * np.cos(Z)
    denominator = 1 + 0.1 * Z**2
    second_input = Z / denominator
    second_input_derivative = (1 - 0.1 * Z**2) / denominator**2
    first_term = np.arctan(np.tan(exp_value))
    second_argument = _safe_tan(second_input)
    derivative = (
        exp_value * exponent_derivative * np.cos(second_argument)
        - first_term
        * np.sin(second_argument)
        * (1 + second_argument**2)
        * second_input_derivative
    )
    derivative[(Z * np.sin(Z) < -ACTIVATION_EXP_LIMIT)
               | (Z * np.sin(Z) > ACTIVATION_EXP_LIMIT)] = 0
    derivative[~np.isfinite(derivative)] = 0
    return dA * derivative


def cubic_exp_tan(Z):
    phase = Z**3 * np.exp(np.cos(Z**2))
    denominator = 1 + np.abs(_safe_tan(np.sin(Z)))
    return np.sin(phase) / denominator


def cubic_exp_tan_backward(dA, Z):
    sin_Z = np.sin(Z)
    tan_sin_Z = _safe_tan(sin_Z)
    exp_phase = np.exp(np.cos(Z**2))
    phase = Z**3 * exp_phase
    phase_derivative = exp_phase * (3 * Z**2 - 2 * Z**4 * np.sin(Z**2))
    denominator = 1 + np.abs(tan_sin_Z)
    denominator_derivative = (
        np.sign(tan_sin_Z) * (1 + tan_sin_Z**2) * np.cos(Z)
    )
    derivative = (
        np.cos(phase) * phase_derivative / denominator
        - np.sin(phase) * denominator_derivative / denominator**2
    )
    derivative[~np.isfinite(derivative)] = 0
    return dA * derivative


FVOX_EXP_LIMIT = 5.0
NROOT_OUTPUT_LIMIT = 2.0
NROOT_DENOMINATOR_EPSILON = 1e-6


def fvonx(Z):
    exp_z = np.exp(np.clip(Z, -FVOX_EXP_LIMIT, FVOX_EXP_LIMIT))
    return np.sin(exp_z)


def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

def fvonx_backward(dA, Z):
    clipped_Z = np.clip(Z, -FVOX_EXP_LIMIT, FVOX_EXP_LIMIT)
    exp_z = np.exp(clipped_Z)
    derivative = exp_z * np.cos(exp_z)
    derivative[(Z < -FVOX_EXP_LIMIT) | (Z > FVOX_EXP_LIMIT)] = 0
    return dA * derivative


def nroot(Z):
    """Evaluate nroot(2^(tan(exp(Z))*cos(Z)), sin(Z)) stably."""
    clipped_Z = np.clip(Z, -FVOX_EXP_LIMIT, FVOX_EXP_LIMIT)
    exp_Z = np.exp(clipped_Z)
    numerator = np.tan(exp_Z) * np.cos(clipped_Z)
    denominator = np.sin(clipped_Z)
    denominator = np.where(
        np.abs(denominator) < NROOT_DENOMINATOR_EPSILON,
        np.where(denominator < 0, -NROOT_DENOMINATOR_EPSILON,
                 NROOT_DENOMINATOR_EPSILON),
        denominator,
    )
    log_output = np.log(2) * numerator / denominator
    return np.exp(np.clip(log_output, -NROOT_OUTPUT_LIMIT, NROOT_OUTPUT_LIMIT))


def nroot_backward(dA, Z):
    clipped_Z = np.clip(Z, -FVOX_EXP_LIMIT, FVOX_EXP_LIMIT)
    exp_Z = np.exp(clipped_Z)
    tan_exp_Z = np.tan(exp_Z)
    sin_Z = np.sin(clipped_Z)
    cos_Z = np.cos(clipped_Z)
    denominator = np.where(
        np.abs(sin_Z) < NROOT_DENOMINATOR_EPSILON,
        np.where(sin_Z < 0, -NROOT_DENOMINATOR_EPSILON,
                 NROOT_DENOMINATOR_EPSILON),
        sin_Z,
    )
    numerator = tan_exp_Z * cos_Z
    numerator_derivative = (
        (1 + tan_exp_Z**2) * exp_Z * cos_Z
        - tan_exp_Z * sin_Z
    )
    log_output = np.log(2) * numerator / denominator
    log_output_derivative = np.log(2) * (
        numerator_derivative * denominator - numerator * cos_Z
    ) / denominator**2
    output = nroot(clipped_Z)
    output_derivative = output * log_output_derivative
    output_derivative[
        (log_output <= -NROOT_OUTPUT_LIMIT)
        | (log_output >= NROOT_OUTPUT_LIMIT)
        | (Z < -FVOX_EXP_LIMIT)
        | (Z > FVOX_EXP_LIMIT)
    ] = 0
    output_derivative[~np.isfinite(output_derivative)] = 0
    return dA * output_derivative


def custom_activation(Z):
    phase = np.sqrt(2) * Z + np.cos(np.pi * Z)
    log_argument = 1 + Z**2
    return (
        3
        + np.tanh(Z) * np.sin(phase)
        + np.cos(np.log(log_argument)) * np.sin(np.sqrt(5) * Z)
    )


def custom_activation_backward(dA, Z):
    phase = np.sqrt(2) * Z + np.cos(np.pi * Z)
    phase_derivative = np.sqrt(2) - np.pi * np.sin(np.pi * Z)
    log_argument = 1 + Z**2
    log_derivative = 2 * Z / log_argument
    derivative = (
        (1 - np.tanh(Z) ** 2) * np.sin(phase)
        + np.tanh(Z) * np.cos(phase) * phase_derivative
        - np.sin(np.log(log_argument)) * log_derivative
        * np.sin(np.sqrt(5) * Z)
        + np.cos(np.log(log_argument)) * np.sqrt(5)
        * np.cos(np.sqrt(5) * Z)
    )
    return dA * derivative


def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    
    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
    elif activation == "softmax":
        activation_func = softmax
    elif activation == "fvonx":
        activation_func = fvonx
    elif activation == "nroot":
        activation_func = nroot
    elif activation == "custom":
        activation_func = custom_activation
    elif activation == "swish_gaussian":
        activation_func = swish_gaussian
    elif activation == "softplus_wave":
        activation_func = softplus_wave
    elif activation == "peir":
        activation_func = peir
    elif activation == "rho1":
        activation_func = rho1
    elif activation == "gelu":
        activation_func = gelu
    elif activation == "mish":
        activation_func = mish
    elif activation == "softsign":
        activation_func = softsign
    elif activation == "arctan_wave":
        activation_func = arctan_wave
    elif activation == "sin":
        activation_func = sin
    elif activation == "wrapped_exp_cos":
        activation_func = wrapped_exp_cos
    elif activation == "cubic_exp_tan":
        activation_func = cubic_exp_tan
    else:
        raise Exception('Non-supported activation function')
        
    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr
        
        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    return A_curr, memory

    return A_curr, memory

def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    epsilon = np.finfo(float).eps
    Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def convert_prob_into_class(Y_hat):
    return (Y_hat >= 0.5).astype(int)

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()


def get_multiclass_cost_value(Y_hat, Y):
    epsilon = np.finfo(float).eps
    return -np.mean(np.sum(Y * np.log(np.clip(Y_hat, epsilon, 1)), axis=0))


def get_multiclass_accuracy_value(Y_hat, Y):
    return np.mean(np.argmax(Y_hat, axis=0) == np.argmax(Y, axis=0))

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]
    
    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    elif activation == "softmax":
        backward_activation_func = softmax_backward
    elif activation == "fvonx":
        backward_activation_func = fvonx_backward
    elif activation == "nroot":
        backward_activation_func = nroot_backward
    elif activation == "custom":
        backward_activation_func = custom_activation_backward
    elif activation == "swish_gaussian":
        backward_activation_func = swish_gaussian_backward
    elif activation == "softplus_wave":
        backward_activation_func = softplus_wave_backward
    elif activation == "peir":
        backward_activation_func = peir_backward
    elif activation == "sin":
        backward_activation_func = sin_backward
    elif activation == "rho1":
        backward_activation_func = rho1_backward
    elif activation == "gelu":
        backward_activation_func = gelu_backward
    elif activation == "mish":
        backward_activation_func = mish_backward
    elif activation == "softsign":
        backward_activation_func = softsign_backward
    elif activation == "arctan_wave":
        backward_activation_func = arctan_wave_backward
    elif activation == "wrapped_exp_cos":
        backward_activation_func = wrapped_exp_cos_backward
    elif activation == "cubic_exp_tan":
        backward_activation_func = cubic_exp_tan_backward
    else:
        raise Exception('Non-supported activation function')
    
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
   
    epsilon = np.finfo(float).eps
    Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values


def full_backward_propagation_multiclass(
    Y_hat, Y, memory, params_values, nn_architecture
):
    grads_values = {}
    dZ_curr = Y_hat - Y

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        A_prev = memory["A" + str(layer_idx_prev)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        dW_curr = np.dot(dZ_curr, A_prev.T) / A_prev.shape[1]
        db_curr = np.mean(dZ_curr, axis=1, keepdims=True)
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

        if layer_idx_prev:
            dA_prev = np.dot(W_curr.T, dZ_curr)
            previous_layer = nn_architecture[layer_idx_prev - 1]
            Z_prev = memory["Z" + str(layer_idx_prev)]
            if previous_layer["activation"] == "relu":
                dZ_curr = relu_backward(dA_prev, Z_prev)
            elif previous_layer["activation"] == "custom":
                dZ_curr = custom_activation_backward(dA_prev, Z_prev)
            elif previous_layer["activation"] == "fvonx":
                dZ_curr = fvonx_backward(dA_prev, Z_prev)
            elif previous_layer["activation"] == "nroot":
                dZ_curr = nroot_backward(dA_prev, Z_prev)
            elif previous_layer["activation"] == "swish_gaussian":
                dZ_curr = swish_gaussian_backward(dA_prev, Z_prev)
            elif previous_layer["activation"] == "softplus_wave":
                dZ_curr = softplus_wave_backward(dA_prev, Z_prev)
            elif previous_layer["activation"] == "peir":
                dZ_curr = peir_backward(dA_prev, Z_prev)
            elif previous_layer["activation"] == "sin":
                dZ_curr = sin_backward(dA_prev, Z_prev)
            elif previous_layer["activation"] == "rho1":
                dZ_curr = rho1_backward(dA_prev, Z_prev)
            elif previous_layer["activation"] == "gelu":
                dZ_curr = gelu_backward(dA_prev, Z_prev)
            elif previous_layer["activation"] == "mish":
                dZ_curr = mish_backward(dA_prev, Z_prev)
            elif previous_layer["activation"] == "softsign":
                dZ_curr = softsign_backward(dA_prev, Z_prev)
            elif previous_layer["activation"] == "arctan_wave":
                dZ_curr = arctan_wave_backward(dA_prev, Z_prev)
            elif previous_layer["activation"] == "wrapped_exp_cos":
                dZ_curr = wrapped_exp_cos_backward(dA_prev, Z_prev)
            elif previous_layer["activation"] == "cubic_exp_tan":
                dZ_curr = cubic_exp_tan_backward(dA_prev, Z_prev)
            else:
                raise Exception(
                    "Unsupported hidden activation for multiclass training"
                )

    return grads_values


def train_multiclass(X, Y, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []
    start_time = time.perf_counter()
    progress_width = 28
    use_live_status = sys.stdout.isatty()

    for i in range(epochs):
        Y_hat, memory = full_forward_propagation(X, params_values, nn_architecture)
        cost = get_multiclass_cost_value(Y_hat, Y)
        accuracy = get_multiclass_accuracy_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy_history.append(accuracy)

        grads_values = full_backward_propagation_multiclass(
            Y_hat, Y, memory, params_values, nn_architecture
        )
        params_values = update(
            params_values, grads_values, nn_architecture, learning_rate
        )
        elapsed = time.perf_counter() - start_time
        completed = i + 1
        progress = completed / epochs
        filled = int(progress_width * progress)
        progress_bar = "=" * filled + "." * (progress_width - filled)
        eta = elapsed / completed * (epochs - completed)
        best_accuracy = max(accuracy_history)
        status = (
            f"\r[{progress_bar}] {completed:>4}/{epochs:<4} "
            f"loss {cost:.4f} | acc {accuracy * 100:6.2f}% | "
            f"best {best_accuracy * 100:6.2f}% | "
            f"time {elapsed:6.1f}s | eta {eta:6.1f}s"
        )
        if use_live_status:
            print(status, end="", flush=True)
        elif completed == 1 or completed == epochs or completed % 100 == 0:
            print(status.lstrip("\r"))

    if use_live_status:
        print()

    return params_values, cost_history, accuracy_history

def update(params_values, grads_values, nn_architecture, learning_rate):
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values;

def train(X, Y, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []
    
    for i in range(epochs):
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        if i % 100 == 0:
            print("Epoch Nr.", i, "abgeschlossen. Die aktuelle genauigkeit betraegt:", accuracy, "bei loss:", cost)
       
    print("Yhatmin:", Y_hat.min())
    print("Yhatmax:", Y_hat.max())
    print("Yhatmean", Y_hat.mean())
    for key in params_values:
        print("Key:", key)
        print("paramsvaluekey:", params_values[key])
    return params_values, cost_history, accuracy_history

def tune(X, Y, nn_architecture, epochs, learning_rate, params_values):
    cost_history = []
    accuracy_history = []
    
    for i in range(epochs):
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
    #print("Epoch Nr.",epochs , "abgeschlossen. Die aktuelle genauigkeit betraegt:", accuracy)    
       
    print("Yhatmin:", Y_hat.min())
    print("Yhatmax:", Y_hat.max())
    print("Yhatmean", Y_hat.mean())
    return params_values, cost_history, accuracy_history

def tune_with_synth_data(nn_architecture, epochs, data_size, bias, learning_rate, params_values):
    cost_history = []
    accuracy_history = []
    X_Test, Y_Test = gen_mixed_points(data_size, bias)

    X_Test = X_Test.T
    Y_Test = Y_Test.reshape(1, -1)
    
    for i in range(epochs):
        X, Y = gen_mixed_points(data_size, bias)
        X = X.T
        Y = Y.reshape(1, -1)
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        if i % 100 == 0:
            Y_hat_Test, cashe = full_forward_propagation(X_Test, params_values, nn_architecture)
            cost_test = get_cost_value(Y_hat_Test, Y_Test)
            accuracy_test = get_accuracy_value(Y_hat_Test, Y_Test)
            print("Epoch Nr.",i , "abgeschlossen. Genauigkeit:", accuracy_test, "Loss:", cost_test)    
       
    print("Yhatmin:", Y_hat.min())
    print("Yhatmax:", Y_hat.max())
    print("Yhatmean", Y_hat.mean())
    return params_values, cost_history, accuracy_history
