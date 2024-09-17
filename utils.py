import numpy as np
from scipy import stats
import arviz as az

def rcspline_eval(x, knots=None, nk=5, inclx=False, knots_only=False,
                  spline_type="ordinary", norm=2, rpm=None, pc=False,
                  fractied=0.05):
    
    """
    Evaluate a restricted cubic spline (RCS) basis for a given set of input values.

    Parameters:
    -----------
    x : array-like
        The input values for which the spline basis is to be evaluated.
    
    knots : array-like, optional
        The locations of the knots. If None, knots will be automatically determined
        based on the data in `x`.
    
    nk : int, optional
        The number of knots to use when `knots` is not specified. Default is 5.
    
    inclx : bool, optional
        Whether to include the original x values as the first column in the output basis.
        Default is False.
    
    knots_only : bool, optional
        If True, return only the knots without evaluating the spline basis. Default is False.
    
    spline_type : str, optional
        Type of spline to use. Can be "ordinary" or "integral". Default is "ordinary".
    
    norm : int, optional
        Normalization method for the spline. 
        - 0: No normalization
        - 1: Normalize based on the difference between the last two knots
        - 2: Normalize based on the difference between the first and last knots raised to the power of 2/3.
        Default is 2.
    
    rpm : float, optional
        Replacement value for NaNs in `x`. If None, NaNs will be left in `x`. Default is None.
    
    pc : bool, optional
        Whether to perform principal component analysis (PCA) on the spline basis. If True, the transformed
        spline basis will be returned. Default is False.
    
    fractied : float, optional
        Fraction of tied values allowed before modifying the knot placement. Should be between 0 and 1.
        Default is 0.05.

    Returns:
    --------
    tuple
        If `knots_only` is False, returns a tuple (xx, knots):
            - xx: The evaluated spline basis matrix.
            - knots: The locations of the knots used.
        If `knots_only` is True, returns only the knots.
    
    Raises:
    -------
    ValueError
        If there are fewer than 6 non-missing observations in `x` when `knots` is not specified.
        If `nk` is less than 3.
        If fewer than 3 unique knots can be obtained.

    Notes:
    ------
    The function automatically determines appropriate knot locations if they are not provided.
    It ensures a minimum of 3 knots and adjusts the knot locations based on the data distribution
    and the `fractied` parameter to handle tied values.

    The evaluated spline basis can be used for regression modeling and other statistical analysis
    where flexible, non-linear relationships are needed.

    Examples:
    ---------
    # Example usage
    x = np.linspace(0, 10, 100)
    spline_basis, knots = rcspline_eval(x, nk=4)
    
    # Get knots only
    knots = rcspline_eval(x, nk=4, knots_only=True)
    """
    x = np.asarray(x)
    if knots is None:  # Knot locations unspecified
        xx = x[~np.isnan(x)]
        n = len(xx)
        if n < 6:
            raise ValueError('knots not specified, and < 6 non-missing observations')

        if nk < 3:
            raise ValueError('nk must be >= 3')

        xu = np.unique(np.sort(xx))
        nxu = len(xu)

        if (nxu - 2) <= nk:
            print(f'{nk} knots requested with {nxu} unique values of x. '
                  f'Knots set to {nxu - 2} interior values.')
            knots = xu[1:-1]  # Exclude first and last elements
        else:
            outer = 0.05 if nk > 3 else 0.1
            if nk > 6:
                outer = 0.025

            nke = nk
            firstknot = lastknot = None
            overrideFirst = overrideLast = False

            if 0 < fractied < 1:
                f = np.bincount(xx) / n
                if max(f[1:-1]) < fractied:
                    if f[0] >= fractied:
                        firstknot = min(xx[xx > min(xx)])
                        xx = xx[xx > firstknot]
                        nke -= 1
                        overrideFirst = True
                    if f[-1] >= fractied:
                        lastknot = max(xx[xx < max(xx)])
                        xx = xx[xx < lastknot]
                        nke -= 1
                        overrideLast = True

            if nke == 1:
                knots = [np.median(xx)]
            else:
                if nxu <= nke:
                    knots = xu
                else:
                    p = np.linspace(outer, 1.0 - outer, nke)
                    knots = np.quantile(xx, p)
                    if len(np.unique(knots)) < min(nke, 3):
                        knots = np.quantile(xx, np.linspace(outer, 1.0 - outer, 2 * nke))
                        if firstknot is not None and lastknot is not None:
                            midval = (firstknot + lastknot) / 2. if lastknot else np.median(xx)
                            knots = np.sort(np.concatenate(([firstknot], [midval], [lastknot])))

                        if len(np.unique(knots)) < 3:
                            print("Fewer than 3 unique knots. Frequency table of variable:")
                            print(np.bincount(x))
                            raise ValueError()

                        print(f"Could not obtain {nke} interior knots with default algorithm.\n"
                              f"Used alternate algorithm to obtain {len(np.unique(knots))} knots.")

                if len(xx) < 100:
                    xx = np.sort(xx)
                    if not overrideFirst:
                        knots[0] = xx[4]
                    if not overrideLast:
                        knots[-1] = xx[-5]

            knots = np.concatenate(([firstknot] if firstknot is not None else [], knots,
                                    [lastknot] if lastknot is not None else []))
    else:
        knots = np.sort(np.unique(knots))
    nk = len(knots)

    if nk < 3:
        print("Fewer than 3 unique knots. Frequency table of variable:")
        print(np.bincount(x))
        raise ValueError()

    if knots_only:
        return knots

    if rpm is not None:
        x[np.isnan(x)] = rpm

    xx = np.zeros((len(x), nk - 2))
    knot1, knotnk, knotnk1 = knots[0], knots[-1], knots[-2]
    kd = 1 if norm == 0 else (knotnk - knotnk1 if norm == 1 else (knotnk - knot1) ** (2 / 3))

    power = 4 if spline_type == "integral" else 3

    for j in range(nk - 2):
        xx[:, j] = np.maximum((x - knots[j]) / kd, 0) ** power + \
                   ((knotnk1 - knots[j]) * np.maximum((x - knotnk) / kd, 0) ** power -
                    (knotnk - knots[j]) * np.maximum((x - knotnk1) / kd, 0) ** power) / \
                   (knotnk - knotnk1)

    if power == 4:
        xx = np.hstack((x[:, None], x[:, None] ** 2 / 2, xx * kd / 4))
    elif inclx:
        xx = np.hstack((x[:, None], xx))

    if pc:
        from sklearn.decomposition import PCA
        pca = PCA()
        xx = pca.fit_transform(xx)
        pcparms = {'center': pca.mean_, 'scale': np.sqrt(pca.explained_variance_), 'rotation': pca.components_}
        xx = pca.transform(xx)
        xx = np.array(xx)
        xx.attrs['pcparms'] = pcparms

    return xx, knots


def h(x, knots):
    x = np.asarray(x)
    
    # Evaluate restricted cubic spline
    spline_basis, _ = rcspline_eval(x, knots)
    
    # Calculate sin and cos components
    sin_component = np.sin(2 * np.pi * x / 12)
    cos_component = np.cos(2 * np.pi * x / 12)
    
    # Combine the spline basis, sin, and cos components
    combined = np.column_stack((spline_basis, sin_component, cos_component))
    
    return combined

def RR_hdi_calculator(trace, hdi_prob = .95):
    hdi = az.hdi(trace, hdi_prob=hdi_prob)
    hdi_lower = hdi.sel(hdi="lower").beta.values
    hdi_higher = hdi.sel(hdi="higher").beta.values
    return np.round(np.exp(hdi_lower), 2), np.round(np.exp(hdi_higher), 2)