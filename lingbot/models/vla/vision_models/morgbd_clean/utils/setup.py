from setuptools import setup, find_packages

setup(
    name="train_utils",
    version="0.0.0",
    description="Train Utils",
    long_description="Train Utils",
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    url="",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "oss2",
        "python-json-logger",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
