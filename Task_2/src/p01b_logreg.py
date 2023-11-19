import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    
    # Const
    EPSILON = 1e-5

    # Fit
    Model = LogisticRegression(eps = EPSILON)
    Model.fit(x_train, y_train)

    # Predict
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept = True)
    preds = Model.predict(x_eval)
    util.plot(x_eval, y_eval, Model.theta, '{}.png'.format(pred_path[:-4]))

    # Save predictions
    np.savetxt(pred_path, preds > 0.5, fmt='%d')

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        # Activation Function —— Sigmoid
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        m, n = x.shape

        # Initialize theta
        if self.theta is None:
            self.theta = np.zeros(n)

        # Newton's Method
        while True:
            Theta = self.theta
            # Loss Function
            J = -(1 / m) * (y - sigmoid(x @ Theta)) @ x

            # Hessian Matrix
            x_Theta = x @ Theta
            H = (1 / m) * (sigmoid(x_Theta) @ sigmoid(1 - x_Theta)) * (x.T) @ x
            H_inv = np.linalg.inv(H)

            # Update theta
            self.theta = Theta - H_inv @ J

            # Check convergence
            if np.linalg.norm(self.theta - Theta, ord=1) < self.eps:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        # Activation Function —— Sigmoid
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        preds = sigmoid(x @ self.theta)

        return preds

        # *** END CODE HERE ***
