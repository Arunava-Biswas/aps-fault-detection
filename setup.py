# importing libraries
from setuptools import find_packages, setup
from typing import List


REQUIREMENT_FILE_NAME="requirements.txt"
HYPHEN_E_DOT = "-e ."

# Creating function to make list of library names required for this project
# the '-> List[str]' just tells that f() returns a list of strings
def get_requirements()->List[str]:
    
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
    requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]
    
    # to remove the '-e .' when installing the requirements.txt
    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)
    return requirement_list


# Creating setup for the library where we can state the version of the library
# The find_packages() is used so it can find all the python source code from the 'sensor' folder
# install_requires is for the list of other libraries we require for our project. 
setup(
    name="sensor",
    version="0.0.1",
    author="ArunavaBiswas",
    author_email="arunavabiswas44@gmail.com",
    packages = find_packages(),
    install_requires=get_requirements(),
)