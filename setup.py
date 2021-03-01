#
# Created by maks5507 (me@maksimeremeev.com)
#

from setuptools import setup
import setuptools.command.build_py as build_py


setup_kwargs = dict(
    name='faiss_wrapper',
    version='0.0.1',
    packages=['faiss_wrapper'],
    install_requires=[
        'numpy'
    ],
    setup_requires=[
    ],

    cmdclass={'build_py': build_py.build_py},
)

setup(**setup_kwargs)
