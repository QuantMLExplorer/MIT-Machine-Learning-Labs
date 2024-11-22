# MIT Machine Learning Course - Practical Exercises

This repository contains practical work and exercises completed as part of the **MIT Machine Learning course** . The course covers principles, algorithms, and applications of machine learning, focusing on supervised and reinforcement learning, with applications to images and temporal sequences.

---

## **Contents**

### **1. Perceptron**
- **Description:** Implementation of the perceptron algorithm and its variations, as well as strategies to evaluate and validate learning algorithms.
- **Topics Covered:**
  - **Perceptron algorithm:**  
    Implemented a simple Perceptron algorithm that iteratively updates weights and bias based on classification      errors. It learns linear decision boundaries through gradient-like updates.
  - **Averaged perceptron:**
    Developed an improved version of the Perceptron algorithm that averages the weights across iterations,           providing better generalization and more stability in the learning process.
  - **Evaluation strategies:**  
    - Testing classifiers on unseen data
    - Cross-validation techniques for performance estimation
    - Evaluating learning algorithms using data sources
  
### **2. Feature Transformations**
- **Description:** Implementation of feature transformation techniques and evaluation of their impact on algorithm performance. This includes transforming discrete values, generating polynomial features, and conducting experiments with various datasets.
- **Topics Covered:**
  - **Encoding Discrete Values:**  
    Transforming discrete features using one-hot encoding to create binary-valued feature vectors.  
  - **Polynomial Features:**  
    Generating higher-order polynomial feature representations to capture non-linear patterns.  
  - **Experimentation:**  
    Evaluating the impact of feature transformations and algorithm choices on dataset performance.  
    - **Datasets:**  
      - **AUTO Data:** Assessing feature transformation effects for regression tasks.  
      - **MNIST Data:** Evaluating feature engineering's role in image classification tasks.

### **3. Margin Maximization and Gradient Descent**
- **Description:** Exploration and implementation of techniques for maximizing margins in classification tasks and using gradient descent for optimization. Includes support vector machine (SVM) objectives and gradient computation
- **Topics Covered:**
  - **Margin Definitions and Scoring Functions:**  
   Analyzing margin definitions and maximizing scoring functions SsumSsum​, SminSmin​, and SmaxSmax​ to improve         classifier robustness.
  - **Hinge Loss and SVM Objective:**  
    Implementing the hinge loss function and combining it with regularization to formulate the SVM objective.
  - **Gradient Descent Implementation:**  
    Implementing numerical gradient estimation and gradient descent to minimize objective functions.
  - **Calculating SVM Objective and Gradient:**  
    Writing functions to compute the SVM objective and its gradient for optimization tasks.
  - **Batch SVM Minimization:**  
    Using gradient descent to optimize the SVM objective with batch processing.

### **4. Linear Regression**
- **Description:** Implementing and optimizing linear regression models using gradient-based methods. This involves computing gradients for both basic linear regression and regularized versions like ridge regression, along with implementing stochastic gradient descent for efficient training
- **Topics Covered:**
  - **Gradient Computation:**  
    Writing functions to compute the gradient of the squared-loss objective for linear regression.
  - **Regularization (Ridge Regression):**  
    Adding a regularization term to the linear regression objective to prevent overfitting, resulting in ridge        regression.
  - **Stochastic Gradient Descent (SGD):**  
    Implementing SGD for efficient optimization, especially when working with large datasets.

### **5. Neural Network**
- **Description:** Implementing the building blocks of a neural network, including forward and backward passes through linear and activation modules. This includes gradient computation, using various activation functions (e.g., tanh, ReLU, SoftMax), and optimizing the model using Stochastic Gradient Descent (SGD).
- **Topics Covered:**
  - **Gradient Computation:**  
    Implementing the forward pass for linear layers, where the input activations are transformed into pre-            activations. The backward pass computes the gradients of the weights and biases with respect to the loss          function, enabling the model to update its parameters during training.
  - **Activation Functions:**  
    Implementing the forward and backward passes for various activation functions, which compute the output           activations and their respective gradients. (e.g., tanh, ReLU, SoftMax)
  - **Stochastic Gradient Descent (SGD):**  
    Implementing SGD to optimize the network's weights. This involves selecting a random sample, calculating the      loss, performing the backward pass, and updating the weights based on the gradients.

### **6. Convolutional Neural Network**
- **Description:**  Implementing key concepts of Convolutional Neural Networks (CNNs) with advanced techniques like batch normalization and mini-batch gradient descent. This involves optimizing neural network parameters, handling convolution layers, and using regularization to improve model performance.
- **Topics Covered:**
  - **Batch Normalization:**  
    Normalizing layer outputs to reduce covariate shift and stabilize training.
  - **Mini-Batch Gradient Descent:**  
    Optimizing weights based on randomly selected subsets of data points, combining stochastic and batch gradient     descent.
  - **Convolutional Layers with Regularization**  
    Applying 1D convolutions with ReLU activation to extract features, while using L1/L2 regularizers to enhance      the model's ability to detect patterns like edges.
    
### **7. State Machines and Markov Decision Processes (MDPs)**
- **Description:**  Implementing and working with state machines and Markov Decision Processes (MDPs) to model decision-making problems and optimize policies.
- **Topics Covered:**
  - **State Machines**  
     Implemented state machines for various examples like binary addition, reverser, and RNN-based models.
  - **MDPs**  
    Worked with MDPs to define and optimize policies using methods like value iteration, greedy, and epsilon-         greedy.
  - **Q-Value Iteration**  
     Implemented Q-value iteration to update Q-functions iteratively for policy improvement.
  - **Receding-Horizon Control**:
    Implemented a recursive approach to compute Q-values for finite horizons with a   discount factor.

### **8. Reinforcement Learning**
- **Description:** Developed Q-learning and neural network-based methods to update Q-values and handle large action/state spaces, applying techniques like batch learning and fitted Q iteration.
- **Topics Covered:**
  - **Q-Update:**  
    Implemented the update rule for Q-values, where Q(s,a) is adjusted towards a target value.
  - **Q_Learn:**  
    Defined a function to iteratively update Q-values using the Q-learning algorithm with a specified learning       rate.
  - **Batch Q-Learn:**  
   Implemented batch Q-learning to update Q-values using multiple episodes, combining previously observed           transitions for more stable learning.
  - **Neural Network Q:**  
   Used neural networks to approximate Q-values for large or continuous state/action spaces, employing squared-Bellman error as the loss function.
  - **Fitted Q Iteration (FQ):**
    Applied Fitted Q Iteration to improve stability by training neural networks on batches of experience, using ϵ-greedy exploration for experience generation.

### **9. Recurrent Neural Networks (RNN)**
- **Description:** Implemented Backpropagation Through Time (BPTT) in an RNN class to optimize the parameters of recurrent networks.
- **Topics Covered:**
  - **BPTT Implementation:**  
    Developed the BPTT method for RNNs, utilizing key attributes like weight matrices (self.Wss, self.Wsx, self.Wo) and activation functions.
  - **Weight Matrices and Activations:**  
    Managed the weight matrices, offsets, activation functions, and their derivatives (e.g., self.f1, self.df1) to compute gradients efficiently.
  - **Backpropagation and Gradient Descent:**  
    Incorporated the hidden state (self.hidden_state) and input dimensions (self.input_dim), and applied gradient descent (self.step_size) to optimize model parameters during training.



 
 

