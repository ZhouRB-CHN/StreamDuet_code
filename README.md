# StreamDuet

Thank you for your interest in our work. This repository contains the implementation for our paper. Below, we provide detailed instructions to help you set up and run the code.

## 1. Install Instructions

To run our code, please ensure that Conda is installed. Then, in the repository root directory, execute the following command to set up the environment:

```conda env create -f conda_environment_configuration.yml```

Once the environment is created, activate it by running:

```conda activate StreamDuet```

This will activate the StreamDuet environment.

---

## 2. Run Our Code

To start the server, execute the following command:
```bash
FLASK_APP=backend/remote_backend.py flask run --port=5000
```

Next, navigate to /workspace and run:  
```bash
python entrance.py
``` 

## 3. View Results
After running the system, you can find the results in the /workspace/results folder. This directory contains all the output data, including logs and processed files.


## 4. Comparison Algorithms

Our experimental code integrates the comparison algorithm DDS with our method. The other two comparison algorithms, AWStream and RMM, need to be executed separately. For your convenience, we have included their source codes in the `awstream` and `RMM` folders, respectively. If you are interested in more details about these algorithms, you can refer to the following literature:

- [1] Ben Zhang, Xin Jin, Sylvia Ratnasamy, John Wawrzynek, and Edward A. Lee. 2018. AWStream: Adaptive wide-area streaming analytics. In *Proceedings of the 2018 Conference of the ACM Special Interest Group on Data Communication*, 236–252.  
- [2] Zhihao Liu, Yuanyuan Shang, Timing Li, Guanlin Chen, Yu Wang, Qinghua Hu, and Pengfei Zhu. 2023. Robust multi-drone multi-target tracking to resolve target occlusion: A benchmark. *IEEE Transactions on Multimedia*, 25 (2023), 1462–1476.


