from setuptools import setup, find_packages
setup(name="robot-sim", version="0.1.0", packages=find_packages(), python_requires=">=3.8", install_requires=["numpy", "torch", "PyYAML", "tensorboard"], extras_require={"export": ["onnx", "onnxruntime"], "sim2sim": ["mujoco"]})
