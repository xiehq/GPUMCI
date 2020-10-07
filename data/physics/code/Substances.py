"""A listing of the substances available in GPUMCI."""
import numpy as np

__all__ = ('substances',)


class Substance(object):
    def __init__(self, name, Z, propotion):
        """ Initialize a substance

        Parameters
        ----------
        name : `str`
            The name of the substance
        Z : `int` array-like
            The atomic numbers of the parts of the substance
        propotion : `float` array-like
            Stoichiometric propotion of the atomic parts.
        """
        self.name = str(name)
        self.Z = np.array(Z, dtype=int)
        self.propotion = np.array(propotion, dtype=float) / np.sum(propotion)
        assert self.Z.shape == self.propotion.shape

    def formula(self):
        import xraylib as xrl
        string = ''
        for z, p in zip(self.Z, self.propotion):
            string += '{}{:.6f}'.format(xrl.AtomicNumberToSymbol(z), p)
        return string

    def __repr__(self):
        return 'Substance({}, {}, {})'.format(self.name, self.Z, self.propotion)


substances = []

#http://physics.nist.gov/cgi-bin/Star/compos.pl?refer=ap&matno=104
substances += [Substance('air',
                         [6, 7, 8, 18],
                         [0.000150, 0.784396, 0.210781, 0.004673])]

#http://physics.nist.gov/cgi-bin/Star/compos.pl?refer=ap&matno=276
substances += [Substance('water',
                         [1, 8],
                         [2.0, 1.0])]

# -- Tissue

substances += [Substance('tissue',
                         [1, 6, 7, 8],
                         [0.630467, 0.058171, 0.011680, 0.299682])]

substances += [Substance('tissue-icru44',
                         [1, 6, 7, 8, 11, 15, 16, 17, 19],
                         [0.631251, 0.074425, 0.015169, 0.276590, 0.000544, 0.000605, 0.000585, 0.000353, 0.000480])]

substances += [Substance('tissue-icrp',
                         [1, 6, 7, 8, 11, 12, 15, 16, 17, 19, 20, 26, 30],
                         [0.629981, 0.117747, 0.010816, 0.239902, 0.000299, 0.000033, 0.000261, 0.000378, 0.000230, 0.000310, 0.000035, 0.000005, 0.000003])]

# -- Brain

substances += [Substance('brain',
                         [1, 6, 7, 8, 11, 12, 15, 16, 17, 19, 20, 26, 30],
                         [0.654256, 0.062356, 0.005660, 0.275312, 0.000478, 0.000037, 0.000682, 0.000330, 0.000397, 0.000473, 0.000013, 0.000005, 0.000001])]

substances += [Substance('brain-icru44',
                         [1, 6, 7, 8, 11, 15, 16, 17, 19],
                         [0.643922, 0.073383, 0.009545, 0.270477, 0.000529, 0.000785, 0.000379, 0.000514, 0.000466])]

substances += [Substance('brain-icrp',
                         [1, 6, 7, 8, 11, 12, 15, 16, 17, 19, 20, 26, 30],
                         [0.654256, 0.062356, 0.005660, 0.275312, 0.000478, 0.000037, 0.000682, 0.000330, 0.000397, 0.000473, 0.000013, 0.000005, 0.000001])]

# -- BONE

#From http://physics.nist.gov/cgi-bin/Star/compos.pl?refer=ap&matno=120
substances += [Substance('bone',
                         [1, 6, 7, 8, 12, 15, 16, 20, 30],
                         [0.474890, 0.122032, 0.030435, 0.283118, 0.000919, 0.034407, 0.000997, 0.053187, 0.000016])]

substances += [Substance('bone-icru44',
                         [1, 6, 7, 8, 11, 12, 15, 16, 20],
                         [0.391834, 0.150222, 0.034894, 0.316456, 0.000506, 0.000957, 0.038699, 0.001089, 0.065343])]

substances += [Substance('bone-icrp',
                         [1, 6, 7, 8, 12, 15, 16, 20, 30],
                         [0.474890, 0.122032, 0.030435, 0.283118, 0.000919, 0.034407, 0.000997, 0.053187, 0.000016])]

# -- TYPES OF BRAIN

substances += [Substance('white_matter',
                         [1, 6, 7, 8, 11, 15, 17, 19],
                         [0.637572, 0.090378, 0.008183, 0.261909, 0.000329, 0.000854, 0.000213, 0.000561])]

substances += [Substance('grey_matter',
                         [1, 6, 7, 8, 11, 15, 17, 19],
                         [0.644105, 0.052170, 0.007856, 0.293837, 0.000401, 0.000695, 0.000260, 0.000676])]

# Common metals
substances += [Substance('aluminium', [13], [1])]
substances += [Substance('iron', [26], [1])]

# Catphan
substances += [Substance('CTP486', [6, 8, 1, 7], [5.588, 1.306, 9.554, 0.166])]