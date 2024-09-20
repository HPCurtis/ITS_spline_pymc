Interupted timseries analysis using Restricted Cubic Splines in PyMC

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
