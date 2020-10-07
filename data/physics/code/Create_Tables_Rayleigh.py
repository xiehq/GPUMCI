"""Create the Rayleigh tables needed by GPUMCI."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from xraylib_np_mock import xrl

EPS = 1e-8


def make_rayleigh_table(substance, energyStep, n_energies, sample_points,
                        plot_result=False):
    """Creates pre-computed interaction tables for Rayleigh interaction.

    Parameters
    ----------
    substance : string
        Name of substance, must be readable by xraylib
    energyStep : float
        Step in MeV between each sample energy
    n_energies : int
        Total number of energy levels to use (starting at 0)
    sample_points : int
        Total number of sample points along each dimension.
    """
    # Settings
    use_form_factor = True

    # Testing
    n = n_energies
    m = sample_points   # Rng data points
    p = 10**4           # Interpolation data points

    eps_theta = np.pi / (2*p)
    theta = np.linspace(eps_theta, np.pi - eps_theta, p)
    mu = np.cos(theta)

    data = np.zeros([m, n])

    for j in range(n):
        Ekv = energyStep * (j+1) * 1000.0

        # rho = Differential Rayleigh scattering cross section (cm2/g/sterad)
        if use_form_factor:
            rho = xrl.DCS_Rayl_CP(substance.formula(), Ekv, theta).squeeze()
        else:
            # Differential Rayleigh scattering cross section (cm2/g/sterad)
            rho = xrl.DCS_Thoms(Ekv, theta).squeeze()

        # Integarate d (cos theta)
        cdf = integrate.cumtrapz(rho, mu, initial=0)

        # Normalize cdf
        cdf = np.abs(cdf) / np.max(np.abs(cdf))

        # Ensure cdf is increasing, for interpolation
        cdf += np.linspace(0, EPS, len(cdf))

        data[:, j] = np.interp(np.linspace(EPS, 1-EPS, m), cdf, mu)

    if plot_result:
        plt.figure()
        im = plt.imshow(data, aspect='auto')
        plt.colorbar(im)
        plt.title(substance.name)
        plt.bone()
        plt.show()

    return data

if __name__ == '__main__':
    from Substances import substances
    for material in substances:
        table = make_rayleigh_table(material, 0.001, 146, 1000, True)
