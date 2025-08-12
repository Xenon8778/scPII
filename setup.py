# setup.py
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scPII", # Replace "scPRS-your-username" with your desired package name
    version="0.0.1",
    author="Shreyan Gupta, TAMU", # Replace "Your Name" with your name
    author_email="xenon8778@tamu.edu", # Replace with your email
    description="A short description of your project.", # A brief description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/scPII", # Replace with your GitHub repository URL
    license="MIT", 
    project_urls={
        "Bug Tracker": "https://github.com/your-username/scPRS/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    package_dir={"": "src"}, # Your Python code will be in a 'src' directory
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9", # Specify the minimum Python version
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "networkx",
        "scikit-learn",
        "joblib",
        "matplotlib",
        "tqdm",
    ],
)