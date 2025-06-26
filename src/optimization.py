import numpy as np


def gradient_descent_step(
        model: np.ndarray,
        gradient: np.ndarray,
        learning_rate: float
):
    """
    Performs a single gradient descent step to update the model.

    Args:
        model (np.ndarray): The current velocity model.
        gradient (np.ndarray): The gradient of the misfit function with respect to the model.
        learning_rate (float): The step length (alpha) for the update.

    Returns:
        np.ndarray: The updated velocity model.
    """

    # --- Preconditioning the gradient (optional but recommended) ---
    # This helps to balance the update. A common technique is to normalize
    # the gradient so its maximum absolute value is 1. This prevents
    # extremely large updates that can destabilize the inversion.

    # Add a small epsilon to avoid division by zero if the gradient is all zeros
    epsilon = 1e-9
    max_grad = np.max(np.abs(gradient))

    if max_grad > epsilon:
        # Normalize the gradient
        scaled_gradient = gradient / max_grad
    else:
        # Gradient is zero, no update to be made
        scaled_gradient = gradient

    # --- Apply the gradient descent update rule ---
    # new_model = current_model - learning_rate * scaled_gradient
    updated_model = model - learning_rate * scaled_gradient

    print(f"  Model updated. Max gradient value: {max_grad:.4e}, Learning rate: {learning_rate}")

    return updated_model