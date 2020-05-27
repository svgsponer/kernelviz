import numpy as np


def add_dila_to_kernel(weights, dilation):
    kernel_length = len(weights) + (len(weights) - 1) * dilation
    kernel = np.zeros(kernel_length)
    kernel_length = kernel.shape[0]
    if dilation == 0:
        kernel = weights
    else:
        sup = 0
        for i in range(0, kernel_length, dilation + 1):
            kernel[i] = weights[sup]
            sup += 1
    return kernel


def apply_kernel(ts, kernel, bias, padding, stride):

    # if padding > 0:
    #     _input_length = len(ts)
    #     _X = np.zeros(_input_length + (2 * padding))
    #     _X[padding:(padding + _input_length)] = ts
    #     X = _X

    # Add padding to kernel
    # kernel = add_dila_to_kernel(weights, dilation)

    kernel_length = kernel.shape[0]

    input_length = ts.shape[0]
    length_diff = input_length - kernel_length
    output_length = ((length_diff) // stride) + 1

    output = np.empty(output_length)

    for i in range(0, output_length):
        _sum = bias
        for j in range(0, kernel_length, stride):
            s = kernel[j] * ts[i + j]
            _sum += s
            output[i] = _sum

    return output


def noramlize(ts):
    ts = (ts-ts.mean())/ts.std()
    return ts
