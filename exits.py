#!/usr/local/bin/python3

import numpy as np

max_pow = 30

samples = np.arange(max_pow).reshape(max_pow, 1)
valuations = np.exp2(samples[20:])
data = np.loadtxt('data.csv', delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7))
data = np.swapaxes(data, 0, 1)

stock = data[0]  # % of company owned
pi = data[1]  # % amount of pi
pi_vals = data[2]  # valuation of pi
investment = data[3]  # Amount of personal investment
duration = data[4]  # Total months spent on the company (normalization)
yearly_rate = data[5]  # salary
time_mul = data[6]  # time multiplier

cost = np.swapaxes(np.repeat([investment + time_mul * duration * yearly_rate / 12], valuations.shape[0], axis=0), 0, 1)
# calculate multiplier

stock_out = np.swapaxes(stock * valuations, 0, 1)
pi_out = np.swapaxes(pi * np.maximum(valuations - pi_vals, [0]), 0, 1)

total = np.zeros(stock_out.shape)

for i in range(stock_out.shape[0]):
    other_pi = np.concatenate((pi_out[:i], pi_out[i+1:]), axis=0)
    other_pi = np.sum(other_pi, 0)
    total[i] = np.maximum(stock_out[i] - other_pi, 0) + pi_out[i]

multiplier = total / cost

output = np.concatenate((np.swapaxes(valuations, 0, 1), multiplier, total - cost), axis=0)

np.savetxt('total.csv', output, delimiter=',')
