from setuptools import setup, find_packages

setup(
    name="booker",
    version="0.1.0",
    description="AI-powered book recommendation system",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Dependencies are handled by requirements.txt
    ],
    entry_points={
        'console_scripts': [
            'booker-viz=booker.viz.cli:app',
        ],
    },
) 