from setuptools import (
    setup,
    find_packages,
)


def get_requirements(filenames):
    r_total = []
    for filename in filenames:
        with open(filename) as f:
            r_local = f.read().splitlines()
            r_total.extend(r_local)
    return r_total

setup(
    name='arekit_eval',
    version='0.23.1',
    description='AREkit evaluation module',
    url='https://github.com/nicolay-r/ARElight',
    author='Nicolay Rusnachenko',
    author_email='???',
    license='MIT License',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='natural language processing, relation extraction, sentiment analysis',
    packages=find_packages(),
    package_dir={'src': 'src'},
    install_requires=get_requirements(['dependencies.txt']),
)