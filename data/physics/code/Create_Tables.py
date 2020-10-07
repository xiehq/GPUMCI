"""Create all data tables needed for GPUMCI using materials in Substances."""

import numpy as np

if __name__ == '__main__':
    from Create_Tables_Rayleigh import make_rayleigh_table
    from Create_Tables_Compton import make_compton_table
    from Create_Tables_Attenuation import make_attenuation_table

    from Substances import substances

    max_energy_kv = 146.0

    attenuation_energies = np.linspace(1e-7, 0.2, 1000)
    num_energies = int(max_energy_kv)
    energy_step_kv = int(num_energies / max_energy_kv)

    energy_step_mv = energy_step_kv / 1000.0

    for material in substances:
        print('Making tables for {} ({})'.format(material.name,
                                                 material.formula()))
        rayleigh_table = make_rayleigh_table(material, energy_step_mv, num_energies, 1000, False)
        file_name = '../interaction/rayleigh_{}_ne{}_estep{}kev.txt'.format(material.name, num_energies, energy_step_kv)
        np.savetxt(file_name, rayleigh_table)

        compton_table = make_compton_table(material, energy_step_mv, num_energies, 1000, False)
        file_name = '../interaction/compton_{}_ne{}_estep{}kev.txt'.format(material.name, num_energies, energy_step_kv)
        np.savetxt(file_name, compton_table)

        attenuation_table = make_attenuation_table(material, attenuation_energies, False)
        file_name = '../attenuation/{}.pdata'.format(material.name)
        np.savetxt(file_name, attenuation_table)
