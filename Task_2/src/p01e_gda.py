import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept = False)

    # *** START CODE HERE ***
    
    # Fit a GDA model
    Model = GDA()
    Model.fit(x_train, y_train)

    # Plot
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept = False)
    preds = Model.predict(x_eval)
    util.plot(x_eval, y_eval, Model.theta, '{}.png'.format(pred_path[:-4]))

    # Save predictions
    np.savetxt(pred_path, preds)

    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        
        m, n = x.shape

        # get phi, mu_0, mu_1, sigma
        phi = (y == 1).sum() / m
        mu_0 = x[y == 0].sum(axis = 0) / (y == 0).sum()
        mu_1 = x[y == 1].sum(axis = 0) / (y == 1).sum()
        diff = x.copy()
        diff[y == 0] -= mu_0
        diff[y == 1] -= mu_1
        sigma = (1 / m) * (diff.T) @ diff

        # Calculate the MLF
        sigma_inv = np.linalg.inv(sigma)
        theta = np.linalg.inv(sigma) @ (mu_1 - mu_0)
        theta0 = 0.5 * (((mu_0.T) @ sigma_inv @ mu_0) - ((mu_1.T) @ sigma_inv @ mu_1)) - np.log((1 - phi) / phi)
        theta0 = np.array([theta0])
        theta = np.hstack([theta0, theta])
        self.theta = theta

        return theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        """
        # Sigmoid
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        x = util.add_intercept(x)

        # print(x.shape, '/n')
        # print(self.theta.shape, '/n')

        # x.shape = (100, 4) after add_intercept
        # self.theta.shape = (3,)

        probs = sigmoid(x @ self.theta)
        preds = (probs >= 0.5).astype(np.int)
        return preds
        """
        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        x = util.add_intercept(x)
        probs = sigmoid(x.dot(self.theta))
        preds = (probs >= 0.5).astype(np.int)
        return preds
        # *** END CODE HERE
