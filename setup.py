import setuptools

MODULE_NAME = "DeepDxAnalyzerBreast"
SOURCE_DIRECTORY_NAME = "Analyzer"
packages = [
    p.replace(SOURCE_DIRECTORY_NAME, MODULE_NAME)
    for p in
    setuptools.find_packages(exclude=['tests'])
]

setuptools.setup(
    name=MODULE_NAME,
    version="0.0.1",
    author="Deep Bio Inc.",
    author_email="code@deepbio.co.kr",
    description="Package for inference on whole slide images(WSIs) of multiple organs",
    url="https://github.com/DEEPBIO/DeepDx-Analyzer-Breast-Resection",
    packages=packages,
    # packages=['PNBAnalyzer'],
    package_dir={MODULE_NAME: SOURCE_DIRECTORY_NAME},
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.7',
    package_data={
        "": ["*.tar"]
    },
    # We will not specify any install_requires here because
    # we don't want installation of this package to change any pre-installed packages.
    # Note that you need following packages to run this package
    # For exact package versions needed to reproduce analysis results, see requirements.txt
    # install_requires=[
    # should change
    # ],
)
