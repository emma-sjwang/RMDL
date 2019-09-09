# keras tensorflow RMDL

Code for paper '[RMDL: Recalibrated Multi-instance Deep Learning for Whole Slide Gastric Image Classification](https://www.sciencedirect.com/science/article/pii/S1361841519300842)' accepted by MedIA 2019.

### Introduction
This implement is based on **GPU** (with a minimum memory of 4~5 GB).

This is a Keras (2.2.1) implementation of [RMDL-inference](https://github.com/EmmaW8/RMDL) with backend of Tensorflow (1.10.1). 
The code was tested with Anaconda and Python (2.7.15).
```Shell
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

### Installation

After installing the dependency:    
``` Shell
    pip install pyyaml
    pip install pytz
    pip install tensorboardX==1.4 matplotlib pillow 
    pip install tqdm
    conda install scipy==1.1.0
    conda install -c conda-forge opencv
```

0. Clone the repo:
    ```Shell
    git clone https://github.com/EmmaW8/RMDL.git
    cd RMDL
    ```

1. Install dependencies:
    
   Annoconda environment installation and activation:
   ```Shell
   conda create -n tf27 pip python=2.7
   source activate tf27
   ```
   Tensorflow installation:
   ```Shell
   pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.1-cp27-none-linux_x86_64.whl
   ```
   Keras installation:
   ```Shell
    pip install keras==2.2.1
   ```
   Install dependencies:
   ```Shell
   conda install -c conda-forge opencv
   pip install openslide-python
   pip install numpy==1.14.5
   pip install tqdm
   pip install matplotlib
   pip install scikit-image
   pip install git+https://www.github.com/keras-team/keras-contrib.git
   ```

2. Configure your dataset path in [configure.yaml](https://github.com/EmmaW8/RMDL/blob/master/configure.yaml) with parameter 'data_dir_list'.
    Download the images and network weights from google [drive](https://drive.google.com/open?id=1dHPCzug8bQAVS2Sv2UT0LlmPdZ2cMlTE).
    you can copy your images (end with '.svs') to the **data** folder. 
    You can also change the gpu number in *gpu_list* and define a larger or smaller batch size according to your GPU memory size.
    
3. Run.    
    If you want to run for your self using the provided images, remove the folder Outputs first.
    ```Shell
    sh run.sh
    ```
   The results will be generated in the *Outputs* folder.



### Citation
>@article{wang2019rmdl,
  title={RMDL: Recalibrated Multi-instance Deep Learning for Whole Slide Gastric Image Classification},
  author={Wang, Shujun and Zhu, Yaxi and Yu, Lequan and Chen, Hao and Lin, Huangjing and Wan, Xiangbo and Fan, Xinjuan and Heng, Pheng-Ann},
  journal={Medical Image Analysis},
  pages={101549},
  year={2019},
  publisher={Elsevier}
}
