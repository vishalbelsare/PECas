# todo: license & copywrite

from setuptools import setup

# todo: license, update email, check dependencies

setup(
    name='pecas',
    version='0.0.1',
    author='Adrian Bürger',
    author_email='adrian.buerger@hs-karlsruhe.de',
    packages=['pecas',
              'pecas.pecas'],
    package_dir={'pecas': 'pecas'},
        url='http://github.com/adbuerger/PECas/',
    license='????',
    zip_safe=False,
    description='Parameter estimation using CasADi in python.',
    install_requires=["numpy >= 1.8",
                      "scipy >= 0.13"],
    use_2to3=True,
)
