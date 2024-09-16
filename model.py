import pymc as pm 

def run_mod(dm, df, offset):
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
        trace = pm.sample(nuts_sampler="numpyro")
        pm.compute_log_likelihood(trace)
        return trace, model