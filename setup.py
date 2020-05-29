from setuptools import setup

setup(
    name = 'deepen',
    version = '0.4.0',
    description = 'A library for building and training deep neural networks.',
    url = 'https://github.com/petejh/deepen',
    author = 'Peter J. Hinckley',
    author_email = 'petejh.code@q.com',
    license = 'MIT',
    packages = ['deepen'],
    install_requires = ['h5py', 'numpy'],
    zip_safe = False
    )
