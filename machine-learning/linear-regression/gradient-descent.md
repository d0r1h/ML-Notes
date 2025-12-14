# Gradient Descent

#### **GD Intuition : Mountain and Valley analogy**&#x20;

The basic idea is analogous to navigating a landscape of mountains (high points) and valleys (low points). **Mountains** represent areas of high error or cost in a model and **Valleys** represent areas of low error or cost.  When you're trying to minimize your model's error (or cost), you want to find the lowest point in this landscape, which is analogous to the **valley**.

**The Process of Gradient Descent**

1. **Starting Point**: Imagine you are standing at a random point on this mountain. This point has a certain height, indicating the error of your model at that point.
2. **Observing the Slope**: To decide which direction to move, you look around and observe the slope of the terrain. This is akin to calculating the **gradient**, which tells you the direction of steepest ascent.
3. **Moving Downhill**: Instead of climbing up, you want to go **downhill**. So, you will move in the opposite direction of the gradient. The steeper the slope, the more you will want to descend, which involves taking larger steps when the gradient is steep and smaller steps when it's shallow.
4. **Iterative Progress**: You repeat this process: at each new point, you calculate the gradient and adjust your position accordingly. Each step leads you closer to the lowest point, or the minimum error.

**Key Points in the Analogy**

* **Step Size (Learning Rate)**: The distance you step downhill at each iteration. If too large, you may overshoot and miss the valley; if too small, the process may take too long.
* **Local Minima**: Sometimes, you might find yourself in a local valley that isn’t the lowest one (global minimum). This occurs when the landscape has several dips.
* **Convergence**: The goal is to keep iteratively stepping down until you reach a point that isn’t decreasing anymore, indicating you’ve found a minimum.

#### GD Algorithm

Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost). Gradient descent is best used when the parameters cannot be calculated analytically (e.g. using linear algebra) and must be searched for by an optimization algorithm.

Gradient Descent is an iterative optimization algorithm that minimizes a loss function by updating parameters in the direction opposite to the gradient. The learning rate controls convergence speed and stability, and variants like SGD and Adam improve efficiency and robustness.

Why GD ? If the number of features is small then we can compute the weights directly (Or use OLS method), However, this Normal Equation is computationally expensive and numerically unstable when:

* n (features) is large (O(n³) for inversion),
* X<sup>T</sup>X is singular or ill-conditioned,
* or data doesn’t fit in memory.

That’s where Gradient Descent (GD) comes in — it’s an iterative optimization method that finds the minimum without matrix inversion.&#x20;

MSE cost for linear regression is Convex Function, which means that there are no local minima, only one global minima. And you can start from any point you’ll reach only optimal (global minima) with a good learning rate.&#x20;

To implement the GD we need to calculate the gradient of the cost function with regard to each model parameter (theta), in other words we need to calculate how much the cost function will change if we change the theta just a little bit.&#x20;

**Steps to calculating parameters with GD**&#x20;

1. Linear Regression Model

$$
\hat{y} = wX + b
$$

_Where : w = weight (slope) and b = bias (intercept)_

2. Loss Function (MSE - Mean Squared Error)

$$
J(w,b) = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2
$$

_Where goal is to minimize the cost function as follow :_&#x20;

$$
\min_{w,b} J(w,b)
$$



3. Gradient

The gradient is the vector of partial derivatives, It tells the direction of steepest increase so we move in the opposite direction. The **derivative** of the function gives us the slope at a particular point. This slope indicates how steep the function is and in which direction it is decreasing the most. By calculating the derivative, we identify the **gradient vector**, which points in the direction of the steepest ascent.

$$
\nabla J(w,b) =
\begin{bmatrix}
\frac{\partial J}{\partial w} \ \
\frac{\partial J}{\partial b}
\end{bmatrix}
$$

4. Deriving gradient for Linear Regression

Partial derivative w.r.t weight w and b :

$$
\frac{\partial J}{\partial w}
=
-\frac{2}{n} \sum_{i=1}^{n} X_i \left(y_i - \hat{y}_i\right)
$$

$$
\frac{\partial J}{\partial b}
=
-\frac{2}{n} \sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)
$$



5. Updating parameter GD rule&#x20;

$$
\theta := \theta - \alpha \nabla J(\theta)  \quad \quad  w := w - \alpha \frac{\partial J}{\partial w}
$$

Above α (alpha) is learning rate which controls the step size.&#x20;

**Steps by step Gradient Descent algorithm  :**&#x20;

1. Initialize w,b randomly&#x20;
2. Repeat Until convergence&#x20;
   1. predict y' (target)
   2. compute loss J(w,b)
   3. compute gradient&#x20;
   4. update parameter&#x20;
3. Stop when&#x20;
   1. loss changes very small
   2. Gradient ≈0
   3. Max iteration reached&#x20;

> Learning Rate (α) is the most important parameter in gradient decent algorithm because if  LR is too high then algorithm will diverge and if LR is too low then it will be slow to converge. Ideal to use the adaptive Learning rate schedules, reduce α over time. Also Learning rate controls stability vs speed trade-off.



&#x20;**There are three different GD methods :-**&#x20;

| Type                | Description                                 | Pros                           | Cons                       |
| ------------------- | ------------------------------------------- | ------------------------------ | -------------------------- |
| Batch GD            | Uses all data to compute gradient each step | Stable, converges smoothly     | Slow for large datasets    |
| Stochastic GD (SGD) | Updates weights for each individual sample  | Fast, good for online learning | Noisy convergence          |
| Mini-Batch GD       | Uses subset (batch) per step                | Balanced speed and stability   | Needs tuning of batch size |

Improving GD training ➖

1. **Feature Scaling** : Gradient descent converges much faster when features are on the same scale.&#x20;
2. **Learning Rate Schedules** : Learning rate α controls step size. If LR is too high then algorithm will diverge and if LR is too low then it will be slow to converge. Ideal to use the adaptive Learning rate schedules, reduce α over time.&#x20;
3. **Regularization** : Adding penalties to the cost function improves generalization and prevents over fitting.
4. **Momentum** **based** **optimizer** : Instead of using plain GD, uses ADAM, RMSProp etc,.
5. **Early** **Stopping** : During training, monitor validation loss and stop when it starts increasing → prevents over fitting
6. **Batch** **Normalization** / **Shuffling** : Shuffle mini-batches each epoch to improve convergence stability, this avoids bias in gradient direction due to data ordering.

#### Code \[Gradient Descent]&#x20;







