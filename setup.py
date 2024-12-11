from setuptools import setup, find_packages

setup(
    name='lightweight_chatbot',
    version='0.1',
    packages=find_packages(),  # Automatically find all packages and sub-packages
    install_requires=[
        # Add any dependencies your project needs here
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        # Add other packages
    ],
    include_package_data=True,
)