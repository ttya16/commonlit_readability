# commonlit_readability
Part of my works of 2 weeks' challenge for [CommonLit Readability Prize Competition](https://www.kaggle.com/c/commonlitreadabilityprize)

### Notebooks referenced
* https://www.kaggle.com/maunish/clrp-pytorch-roberta-pretrain
* https://www.kaggle.com/maunish/clrp-pytorch-roberta-finetune

## Environment
* [kaggle docker image](https://github.com/Kaggle/docker-python)
* Docker + [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Setting `docker-compose.yml` as follows may be an easy way to build and run the container.
```docker
version: '3.8'
services:
    clrp:
        build:
            context: clrp
            dockerfile: Dockerfile
        container_name: clrp
        user: root
        environment:
            NVIDIA_VISIBLE_DEVICES: all
        ports:
        - "8889:8888"
        tty: true
        volumes:
            - ./clrp:/clrp
```

## Dataset Download
At the top of the project's directory, create `data` folder.
```
cd .
mkdir data
```
and place datasets for the competition.
```
data
├── sample_submission.csv
├── test.csv
└── train.csv
```
You can easily download datasets using Kaggle API.
```
kaggle competitions download -c commonlitreadabilityprize
```
