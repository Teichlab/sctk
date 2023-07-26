from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='sctk',
    version='0.1.1',
    author='nh3',
    author_email='nh3@users.noreply.github.com',
    description='single cell analysis tool kit based on scanpy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Teichlab/sctk',
    packages=find_packages(),
    entry_points=dict(),
    install_requires=[
        'packaging',
        'scanpy',
        'leidenalg',
        'adjustText',
        'numpy_groupies',
    ],
)
