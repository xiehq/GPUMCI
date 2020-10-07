"""Create the cross-section tables needed by GPUMCI."""

import numpy as np
import matplotlib.pyplot as plt
from xraylib_np_mock import xrl


def make_attenuation_table(material, energies, plot_result=False):
    formula = material.formula()

    energies_mv = np.asarray(energies)
    energies_kv = energies_mv * 1000.0

    data = np.zeros([energies_kv.size, 8])

    data[:, 0] = energies_mv
    data[:, 1] = xrl.CS_Rayl_CP(formula, energies_kv)
    data[:, 2] = xrl.CS_Compt_CP(formula, energies_kv)
    data[:, 3] = xrl.CS_Photo_CP(formula, energies_kv)
    data[:, 4] = 0  # We ignore pair production
    data[:, 5] = 0  # We ignore pair production
    data[:, 6] = xrl.CS_Total_CP(formula, energies_kv)
    data[:, 7] = data[:, 6] - data[:, 1]


    if plot_result:
        plt.figure()
        plt.title(material.name)
        plt.plot(energies_mv, np.log(data))
        plt.show()

    return data


if __name__ == '__main__':
    from Substances import substances

    n_energies = 1000
    energies = np.linspace(1e-7, 0.2, n_energies)

    for material in substances:
        table = make_attenuation_table(material, energies, plot_result=True)
