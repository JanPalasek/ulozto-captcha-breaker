import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ulozto-captcha-breaker',
    version='3.0a',
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={
        "": "src"
    },
    install_requires=[
        "tensorflow>=2.0.0",
        "matplotlib>=3.5.0"
    ],
    packages=setuptools.find_packages("src"),
    scripts=["bin/train.py", "bin/test.py", "bin/predict.py"],
    python_requires=">=3.8"
)