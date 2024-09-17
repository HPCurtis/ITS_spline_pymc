import pymc as pm 

def run_mod(dm, df, offset):
   """
    Runs a Bayesian Poisson regression model using PyMC and returns the sampled trace and model object.

    This function sets up and runs a Poisson regression model using a design matrix `dm`, data frame `df`,
    and an offset term `offset`. It uses  normal priors for the 
    intercept and coefficients, and estimates the posterior distributions of the parameters using the 
    Hamiltonian Monte Carlo (HMC) sampler with Numpyro.

    Parameters:
    -----------
    dm : array-like
        The design matrix containing the predictor variables for the regression model.
    
    df : pandas.DataFrame
        The data frame containing the observed data, including the response variable `rate`.
    
    offset : float or array-like
        The offset term to be added to the linear predictor. This is often used in Poisson regression 
        to account for different exposure times or population sizes.

    Returns:
    --------
    trace : arviz.InferenceData
        The trace containing the posterior samples of the model parameters.
    
    model : pm.Model
        The PyMC model object that was used for the analysis.

    Notes:
    ------
    - The model assumes a Poisson likelihood for the response variable.
    - Priors for the intercept (`alpha`) and regression coefficients (`beta`) are specified as normal distributions.
    - The function uses the NUTS sampler implemented in Numpyro to draw samples from the posterior distribution.
    - The log-likelihood is computed for model comparison purposes.

    Example:
    --------
    dm = np.array([[1, 0], [1, 1], [1, 2]])
    df = pd.DataFrame({'rate': [2, 3, 4]})
    offset = 0.5
    trace, model = run_mod(dm, df, offset)
    """
   with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=dm.shape[1]) 

        # Calculate predicted values from model. 
        mu = pm.Deterministic("mu",
            alpha + offset + 
            pm.math.dot(dm, beta)
        )
        
        # Poisson rate parameter
        lambda_ = pm.Deterministic("lamda", pm.math.exp(mu)) 
        
        # Likelihood
        y = pm.Poisson("y", mu=lambda_, observed=df['rate'])
        
        # Run HMC sampler (Numpyro) to estimate parameter posteriors. 
        trace = pm.sample(nuts_sampler="numpyro")

        # Compute the log-likelihood for model comparison.
        pm.compute_log_likelihood(trace)
        return trace, model