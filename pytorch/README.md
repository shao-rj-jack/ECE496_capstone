## Setup
This setup process uses Python 3.10.8 on a Linux environment. NOTE: CUDA will not install binaries on MACOS, and venv activation is slightly different on Windows. To follow this setup process, use some sort of pytorch supported Linux distribution.

To create a virtual environment to isolate pip dependencies, in the pytorch directory run:

```bash
python3 -m virtualenv env
```

To activate the virtual env, run:

```bash
source env/bin/activate
```

To install all dependencies, run:

```bash
python3 -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
