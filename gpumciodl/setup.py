# -*- coding: utf-8 -*-
"""

"""
from future import standard_library
standard_library.install_aliases()

from distutils.core import setup
import os
import site

install_to = os.path.join(site.getsitepackages()[0],'gpumci','data')
data_files=[(install_to,list(map(lambda y: x[0]+'/'+y, x[2]))) for x in os.walk('../data/physics/')]

print(data_files)

setup(name='gpumci',
      version='0.1.0',
      author='Jonas Adler',
      author_email='jonasadl@kth.se',
      url='https://gits-14.sys.kth.se/jonasadl/GPUMCI',
      description='Python bindings for the GPUMCI library',
      license='Proprietary',
      packages=['gpumci'],
      package_dir={'gpumci': '.'}, 
      data_files=data_files,
      package_data={'gpumci': ['*.*']})
