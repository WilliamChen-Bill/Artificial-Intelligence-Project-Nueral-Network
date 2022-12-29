# Artificial Intelligence Project: Nueral Network
## Assignment Goals
* Get Pytorch set up for your environment.
* Familiarize yourself with the tools.
* Implementing and training a basic neural network using Pytorch.
* Happy deep learning:)
## Summary
* Home-brewing every machine learning solution is not only time-consuming but potentially error-prone. One of the reasons we’re using Python in this course is because it has some very powerful machine learning tools. Besides common scientific computing packages such as SciPy and NumPy, it’s very helpful in practice to use frameworks such as Scikit-Learn, TensorFlow, Pytorch, and MXNet to support your projects. The utilities of these frameworks have been developed by a team of professionals and undergo rigorous testing and verification. In this homework,  we’ll be exploring the [<span style="color: blue">Pytorch</span>](https://pytorch.org) framework. Please complete  the functions in the template provided: <span style="color: red">intro_pytorch.py</span>.
## Part 1: Setting up the Python Virtual Environment
* In this assignment, you will familiarize yourself with the Python Virtual Environment. Working in a virtual environment is an important part of working with modern ML platforms, so we want you to get a flavor of that through this assignment. Why do we prefer virtual environments? Virtual environments allow us to install packages within the virtual environment without affecting the host system setup. So you can maintain project-specific packages in respective virtual environments.
* We suggest that you use the CS lab computers for this homework. You can also work on your personal systemfor the initial development, but finally, you will have to test your model on the CSL lab computers. Find more instructions: [<span style="color:blue">How to access CSL Machines Remotely</span>](https://csl.cs.wisc.edu)
* The following are the installation steps for Linux (CSL machines are recommended). You will be working on Python 3 (instead of Python 2 which is no longer supported). Read more about Pytorch and Python version [<span style = "color:blue">here</span>](https://pytorch.org/get-started/locally/). To check your Python version use:
```bash
python -v or python3 -v
```
* If you have an alias set for python=python3 then both should show the same version(3.x.x)
* **Step 1:** For simplicity, we use the [<span style = "color:blue">venv</span>](https://docs.python.org/3/library/venv.html) module (feel free to use other virtual envs such as [<span style = "color:blue">Conda</span>](https://www.anaconda.com)).To set up a Python Virtual Environment named Pytorch:
```bash
python3 -m venv /path/to/new/virtual/environment
```
* For example, if you want to put the virtual environment in your working directory:
```bash
python3 -m venv Pytorch
```
* (Optional: If you want to learn more about Python virtual environments, a very good tutorial can be found [<span style = "color:blue">here</span>](https://realpython.com/python-virtual-environments-a-primer/)
* **Step 2**: Activate the environment:
* Let’s suppose the name of our virtual environment is Pytorch (you can use any other name if you want). You can activate the environment by the following command:
```bash
source Pytorch/bin/activate
```
