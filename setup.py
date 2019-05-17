from setuptools import setup

with open("requirements.txt") as f:
    reqs = f.read().splitline()

setup(
    name = "mftrees",
    version = "0.0.1",
    author = "Pramukta Kumar",
    author_email = "pramukta@asu.edu",
    description = ("Rainforest carbon estimation from satellite imagery using Fourier power spectra, Manifold embeddings, and xgboost"),
    url = "https://github.com/pramukta/mftrees",
    packages=['mftrees'],
    entry_points={
        "console_scripts": [
            "mft.features = mftrees.features:oneshot",
            "mft.train = mftrees.manifold:main",
            "mft.predict = mftrees.models:predict",
            "mft.histmatch = mftrees.util:histmatch"
        ]
    },
    install_requires=reqs
)
