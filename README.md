# Introduction

Deep Neural Networks (DNNs) are used in many vital applications, such as determining loan grant decisions and serving as the controllers for autonomous cars, due to their effectiveness and efficacy in performing complex tasks. In these circumstances, the DNNs must frequently adhere to certain domain-specific requirements. Monotonicity is one such requirement.

A Deep Neural Network is monotonically increasing with respect to some input features f if the output of the network increases if we increase the values of those features. Similarly, monotonically decreasing would mean that the output of the network decreases with increase in value of the features.

Some real-life examples include:

- Increase in salary should give increased predicted loan amount
- Increase in number of rooms should give increased predicted house price
- Decrease in crime rate of area should give increased predicted house price
- Increase in hours per week should give increased predicted income level

If DNNs are used to solve all the cases mentioned above, then it is important for explainability and reliability that the networks follow the above mentioned requirements. If not, then it can be difficult to trust them.

# Methodology

We implement the Point-wise monotonic loss defined by Gupta et. al. [1] in the [monotonic-training.py](src/monotonic-training.py).

# References

1. [Akhil Gupta, Naman Shukla, Lavanya Marla, Arinbjörn Kolbeinsson, and Kartik Yellepeddi.
How to incorporate monotonicity in deep networks while preserving flexibility?](https://arxiv.org/abs/1909.10662)

