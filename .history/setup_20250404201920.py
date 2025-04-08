from setuptools import setup, find_packages

setup(
    name="crypto_btc",
    version="0.1",
    packages=find_packages(),
)


from setuptools import setup, find_packages

setup(
    name="Bitcoin",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"}
    
)