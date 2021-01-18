import setuptools

setuptools.setup(
    name='DrC-Net',
    version='1.0.0',
    description='Unsupervised deep learning for susceptibility induced distortion correction in diffusion MRI',
    url='https://github.com/YuchuanQiao/DrC-Net',
    keywords=['distortion', 'correction'],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache License 2.0',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'sklearn',
        'nibabel',
        'matplotlib',
        'pprint',
        'tqdm',
    ]
)
