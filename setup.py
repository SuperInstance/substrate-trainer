from setuptools import setup, find_packages

setup(
    name="substrate-trainer",
    version="0.1.0",
    description="Train JEPA-like models on the substrate's witness log. The substrate is the soil. The model is the plant. The witness log is the rain.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="SuperInstance",
    license="MIT",
    py_modules=["trainer"],
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "substrate-trainer=trainer:_cli",
        ],
    },
)
