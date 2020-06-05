import numpy as np

def predict(beta, x):
    return np.matmul(x, beta)

def error(beta, x, y):
    return (y - predict(beta, x))

def update_beta(gradient_fn, beta, *args, rate=1e-6):
    return beta - rate * gradient_fn(beta, *args)

# Use uniform step size (learning rate)
def gradient_descent(beta, gradient_fn, *args, rate=1e-6, tol=1e-6, verbose=False, ret_intermediate_betas=False):
    tol = np.array([tol] * len(beta))
    beta_next = update_beta(gradient_fn, beta, *args, rate=rate)
    if ret_intermediate_betas:
        betas = [beta]
    diff = abs(beta_next - beta)
    iters = 1
    while (diff > tol).any():
        beta = beta_next
        if ret_intermediate_betas:
            betas.append(beta)
        beta_next = update_beta(gradient_fn, beta, *args, rate=rate)
        diff = abs(beta_next - beta)
        iters += 1
        if verbose and (iters % 5000 == 0):
            print('Iteration %d. Mean difference = %f' % (iters, diff.mean()))
    if ret_intermediate_betas:
        betas.append(beta_next)
        return np.array(betas)
    return beta_next

