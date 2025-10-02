from setuptools import find_packages, setup

setup(
    name="resume-optimizer-agent",
    version="0.1.0",
    description="AI-powered resume optimizer and job harvester",
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "typer[all]>=0.9",
        "rich>=13.7",
        "pydantic>=1.10",
        "requests>=2.32",
        "python-dotenv>=1.0",
    ],
    entry_points={
        "console_scripts": [
            "resume-opt=cli.resume_opt:app",
        ]
    },
)
