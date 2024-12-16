from setuptools import setup, find_packages

setup(
    name="neural_gui",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
        "PyQt6>=6.4.0",
        "tensorflow-cpu>=2.15.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="GUI приложение для работы с нейронными сетями",
    keywords="neural networks, deep learning, PyQt6",
    python_requires=">=3.8",
) 