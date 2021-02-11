"""Alpacka installation script."""

import setuptools


setuptools.setup(
    name='alpacka',
    description='AwareLab PACKAge - internal RL framework',
    version='0.0.1',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'cloudpickle',
        'gin-config',
        'gym[atari]',
        'joblib',
        'numpy',
        'scipy',
        'tblib',
        'tensorflow>=2.2.0',
    ],
)
