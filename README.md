# Interrupted timseries analysis using Restricted Cubic Splines in PyMC

## File structure
- README.md: The file your reading right now.
- its.ipynb: Jupyter notebook containing the python code for the analysis of the dataset.
- its.pdf: pdf form of ipython notebook.
- model.py: python file containing the Posoon regression PyMC model code used in the its.ipynb notebook.
- utils.py: python file of variosu utility fucntionf or the anlaysis used in the its.ipynb notebook. The file contains eh code for genration of the RCS splines cyclical model components and Risk Ration calculations.
- vis 
  - ACE_scatter.png: PNG file plot showing the ACE rate for sicily between 2002-2006.
  - rcs_fit.png: PNG file fro model fit for RCS model. 
  - cyl_fit.png: PNG file fro model fit for RCS + Cyl model.
  - PPC_rcs.png: PNG file showing the posterior predictive plots produced by the arviz package for RCS model
  - PPC_cyl.png: PNG file showing the posterior predictive plots produced by the arviz package for RCS + Cyl model.
  - rank_rcs.png: PNG file showing the rank plots for MCMC chains produced by the arviz package for RCS model.
  - rank_cyl.png: PNG file showing the rank plots for MCMC chains produced by the arviz package for RCS + Cyl model.
  - loo_comparison.png: PNG file contianing a plot showing the outcome of LOO-Cv mdoel comparsion for RCS and RCS + Cyl models.


# References

Abril-Pla, O., Andreani, V., Carroll, C., Dong, L., Fonnesbeck, C. J., Kochurov, M., ... & Zinkov, R. (2023). PyMC: a modern, and comprehensive probabilistic programming framework in Python. PeerJ Computer Science, 9, e1516.

Barone-Adesi, F., Gasparrini, A., Vizzini, L., Merletti, F., & Richiardi, L. (2011). Effects of Italian smoking regulation on rates of hospital admission for acute coronary events: a country-wide study. PloS one, 6(3), e17419.

Bernal, J. L., Cummins, S., & Gasparrini, A. (2017). Interrupted time series regression for the evaluation of public health interventions: a tutorial. International journal of epidemiology, 46(1), 348-355.

Bhaskaran, K., Gasparrini, A., Hajat, S., Smeeth, L., & Armstrong, B. (2013). Time series regression studies in environmental epidemiology. International journal of epidemiology, 42(4), 1187-1195.

Brodersen, K. H., Hauser, A., & Hauser, M. A. (2017). Package CausalImpact. Google LLC: Mountain View, CA, USA.

Lundberg, I., Johnson, R., & Stewart, B. M. (2021). What is your estimand? Defining the target quantity connects statistical evidence to theory. American Sociological Review, 86(3), 532-565.

McElreath R. Statistical Rethinking. 2nd Edition. London (UK): Routledge; 2020.

Stone, C. J., & Koo, C. Y. (1985). Additive splines in statistics. Proceedings of the Statistical Computing Section ASA, 45–48.

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021). Rank-normalization, folding, and localization: An improved R ̂ for assessing convergence of MCMC (with discussion). Bayesian analysis, 16(2), 667-718.

# Web resources
https://hbiostat.org/rmsc/

https://github.com/harrelfe/Hmisc