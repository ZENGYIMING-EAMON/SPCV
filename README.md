# SPCV

![Beam Image](https://github.com/ZENGYIMING-EAMON/SPCV/blob/main/images/beam_image.png?raw=true)

![GIF](https://github.com/your-username/your-repository/blob/main/images/animation.gif?raw=true)

This repository includes the source code of the paper [Dynamic 3D Point Cloud Sequences as 2D Videos](https://arxiv.org/abs/2403.01129).

Authors: [Yiming Zeng](https://scholar.google.com/citations?user=1BSTaEUAAAAJ&hl=zh-TW), [Junhui Hou](https://sites.google.com/site/junhuihoushomepage/), [Qijian Zhang](https://keeganhk.github.io/), [Siyu Ren](https://scholar.google.com/citations?user=xSm7_VwAAAAJ&hl=en), [Wenping Wang](https://engineering.tamu.edu/cse/profiles/Wang-Wenping.html).

###  <a href="https://arxiv.org/pdf/2403.01129" target="_blank">Paper PDF</a> </a>

### Install all dependencies  
```shell
git clone https://github.com/ZENGYIMING-EAMON/SPCV.git
cd SPCV 
conda create -n SPCV python=3.8
conda activate SPCV

# install pytorch (https://pytorch.org/)
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 --extra-index-url https://download.pytorch.org/whl/cu111

# install pytorch3d 0.7.3 
pip install "git+https://github.com/facebookresearch/pytorch3d.git@e245560abb8f019a24880faf7557ed3b2eec6cc0"

# install other dependencies
conda env update --file environment.yml
pip install -r requirements.txt 
```

<details>
  <summary> Pip Dependencies (click to expand) </summary>

absl-py==1.4.0
addict==2.4.0
aiohttp==3.8.1
aiosignal==1.3.1
asttokens==2.2.1
async-timeout==4.0.2
attrs==23.1.0
backcall==0.2.0
blessed==1.20.0
cachetools==5.3.0
certifi==2023.5.7
cffi==1.15.1
chamferdist==1.0.0
charset-normalizer @ file:///home/conda/feedstock_root/build_artifacts/charset-normalizer_1678108872112/work
click==8.1.3
colorama @ file:///home/conda/feedstock_root/build_artifacts/colorama_1666700638685/work
comm==0.1.3
ConfigArgParse==1.5.3
contourpy==1.0.7
cryptography==40.0.2
cycler==0.11.0
dash==2.9.3
dash-core-components==2.0.0
dash-html-components==2.0.0
dash-table==5.0.0
debugpy==1.6.7
decorator==5.1.1
Deprecated==1.2.13
drjit==0.4.2
easydict==1.10
einops==0.6.0
emd-ext==0.0.0
et-xmlfile==1.1.0
executing==1.2.0
fastjsonschema==2.16.3
filelock==3.11.0
Flask==2.2.3
fonttools==4.39.3
frozenlist==1.3.3
future @ file:///home/conda/feedstock_root/build_artifacts/future_1673596611778/work
fvcore @ file:///home/conda/feedstock_root/build_artifacts/fvcore_1671623667463/work
github==1.2.7
google-auth==2.18.1
google-auth-oauthlib==1.0.0
gpustat==1.1
grpcio==1.54.2
h5py==3.8.0
huggingface-hub==0.13.4
idna @ file:///home/conda/feedstock_root/build_artifacts/idna_1663625384323/work
imageio==2.27.0
importlib-metadata==6.4.1
importlib-resources==5.12.0
iopath==0.1.10
ipdb==0.13.13
ipykernel==6.22.0
ipython==8.12.0
ipywidgets==8.0.6
itsdangerous==2.1.2
jedi==0.18.2
Jinja2==3.1.2
joblib @ file:///home/conda/feedstock_root/build_artifacts/joblib_1663332044897/work
jsonpatch==1.32
jsonpointer==2.3
jsonschema==4.17.3
jupyter_client==8.2.0
jupyter_core==5.3.0
jupyterlab-widgets==3.0.7
kiwisolver==1.4.4
kornia @ git+https://github.com/kornia/kornia@8979f4a45d05f1f56f9c28e23870699a914805f0
lazy_loader==0.2
loguru @ file:///croot/loguru_1675318478402/work
Markdown==3.4.3
markdown-it-py==3.0.0
MarkupSafe==2.1.2
matplotlib==3.7.1
matplotlib-inline==0.1.6
mdurl==0.1.2
meshio==5.3.4
mitsuba==3.3.0
multidict==6.0.4
nbformat==5.7.0
nest-asyncio==1.5.6
networkx==3.1
neuralnet-pytorch==0.0.3
numpy @ file:///home/conda/feedstock_root/build_artifacts/numpy_1651020413938/work
nvidia-ml-py==11.525.112
oauthlib==3.2.2
olefile @ file:///home/conda/feedstock_root/build_artifacts/olefile_1602866521163/work
open3d==0.17.0
opencv-python==4.7.0.72
openpyxl==3.1.2
ordered-set==4.1.0
packaging==23.1
pandas==2.0.0
parso==0.8.3
pdf2image==1.16.3
pexpect==4.8.0
pickleshare==0.7.5
Pillow==9.5.0
pkgutil_resolve_name==1.3.10
platformdirs==3.2.0
plotly==5.14.1
plyfile==0.9
point-cloud-utils==0.29.3
pointnet2==0.0.0
pointnet2-ops @ git+https://github.com/erikwijmans/Pointnet2_PyTorch.git@b5ceb6d9ca0467ea34beb81023f96ee82228f626#subdirectory=pointnet2_ops_lib
portalocker @ file:///home/conda/feedstock_root/build_artifacts/portalocker_1674135640384/work
prompt-toolkit==3.0.38
protobuf==3.20.3
psutil==5.9.5
ptyprocess==0.7.0
pure-eval==0.2.2
pyasn1==0.5.0
pyasn1-modules==0.3.0
pycparser @ file:///home/conda/feedstock_root/build_artifacts/pycparser_1636257122734/work
PyGithub==1.58.1
Pygments==2.15.0
PyJWT==2.6.0
pykdtree==1.3.7.post0
PyLaTeX==1.4.1
pymeshlab==2022.2.post4
PyNaCl==1.5.0
pyparsing @ file:///home/conda/feedstock_root/build_artifacts/pyparsing_1652235407899/work
pyquaternion==0.9.9
pyrsistent==0.19.3
PySocks @ file:///home/conda/feedstock_root/build_artifacts/pysocks_1661604839144/work
python-dateutil==2.8.2
pytz==2023.3
PyWavelets==1.4.1
PyYAML @ file:///home/conda/feedstock_root/build_artifacts/pyyaml_1648757091578/work
pyzmq==25.0.2
requests @ file:///home/conda/feedstock_root/build_artifacts/requests_1684774241324/work
requests-oauthlib==1.3.1
rich==13.4.2
rsa==4.9
scikit-image==0.20.0
scikit-learn==1.2.2
scipy==1.10.1
six==1.16.0
stack-data==0.6.2
tabulate @ file:///home/conda/feedstock_root/build_artifacts/tabulate_1665138452165/work
tenacity==8.2.2
tensorboard==2.13.0
tensorboard-data-server==0.7.0
tensorboardX==2.6
termcolor @ file:///home/conda/feedstock_root/build_artifacts/termcolor_1672833821273/work
threadpoolctl @ file:///home/conda/feedstock_root/build_artifacts/threadpoolctl_1643647933166/work
tifffile==2023.4.12
timm==0.6.13
tomli==2.0.1
torch-geometric @ file:///usr/share/miniconda/envs/test/conda-bld/pyg_1679555056114/work
torch-scatter==2.1.1
torch-sparse==0.6.12
tornado==6.3
tqdm @ file:///home/conda/feedstock_root/build_artifacts/tqdm_1677948868469/work
traitlets==5.9.0
transforms3d==0.4.1
trimesh==3.21.5
typing_extensions @ file:///home/conda/feedstock_root/build_artifacts/typing_extensions_1685704949284/work
tzdata==2023.3
unfoldNd==0.2.0
unrar==0.4
urllib3 @ file:///home/conda/feedstock_root/build_artifacts/urllib3_1686156552494/work
vedo==2023.4.6
visdom==0.2.4
vtk==9.0.3
wcwidth==0.2.6
websocket-client==1.5.2
Werkzeug==2.2.3
widgetsnbextension==4.0.7
wrapt==1.15.0
yacs @ file:///home/conda/feedstock_root/build_artifacts/yacs_1645705974477/work
yarl==1.8.2
zipp==3.15.0

</details>


### Quick start 
```
bash ./run.sh
```


### Generate Data for Training

The process has been integrated into the `run.sh` script. You can change the name `swing_pick_objs` in these four Python scripts to point the pipeline to your own mesh sequences folder.

If you want to start from raw point cloud sequences, you can omit the mesh folder in these scripts and start from the folder `xxx_objs_sample1w`, then scale and process them into `xxx_objs_sample1w_fpsUnify`.



### Citation 
If you find our code or paper helps, please consider citing:
```
@article{zeng2024dynamic,
  title={Dynamic 3D Point Cloud Sequences as 2D Videos},
  author={Zeng, Yiming and Hou, Junhui and Zhang, Qijian and Ren, Siyu and Wang, Wenping},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
}
``` 