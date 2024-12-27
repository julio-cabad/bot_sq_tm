from setuptools import setup, find_packages

setup(
    name="tm_squeeze",
    packages=find_packages(include=[
        'strategies', 'exchange', 'backtesting', 
        'indicators', 'utils', 'config'
    ]),
    version="0.1.0",
) 