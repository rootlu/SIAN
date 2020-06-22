# SIAN
Code and data for ECML-PKDD paper "[Social Influence Attentive Neural Network for Friend-Enhanced Recommendation](https://yuanfulu.github.io/publication/PKDD-SIAN.pdf)"

## Requirements

- Python 2.7
- PyTorch 0.4.1
- numpy
- scipy
- My machine with two GPUs (NVIDIA GTX-1080 *2) and two CPUs (Intel Xeon E5-2690 * 2)

## Description

```
├── baselines  # baseline code
│   ├── Eval4Baselines.py

├── code  # Our Model: SIAN
│   ├── Attention.py  # attention layer
│   ├── DataUtil.py  # data loader 
│   ├── Evaluation.py  # model evaluation
│   ├── FeatureAgg.py  # attentive feature aggregator 
│   ├── Fusion.py  # feature fusion layer 
│   ├── HeteInf.py  # the main class for SIAN
│   ├── InfluenceProp.py  # social influence coupler 
│   ├── Logging.py  #log

│   └── trainHeteInf.py  # the main function for SIAN
└── data  # dataset
    ├── Data4Baselines.ipynb  #
    ├── DataProcessor.ipynb
    ├── ItemProfileEmbed.ipynb
    ├── WechatTencent.ipynb
    ├── wxt  # FWD dataset 
    │   ├── biz2id
    │   ├── biz_profile.npy
    │   ├── item2id
    │   ├── item_profile.npy
    │   ├── user2id
    │   ├── user_profile.npy
    │   ├── wxt.att.analysis
    │   ├── wxt.interaction.graph
    │   ├── wxt.item.biz
    │   ├── wxt.social.graph
    │   ├── wxt.test.rating.712
    │   ├── wxt.train.rating.712
    │   ├── wxt.user.biz
    │   └── wxt.val.rating.712
    ├── wxt.ipynb
    ├── yelp  # yelp dataset
    │   ├── item_profile.npy
    │   ├── user_profile.npy
    │   ├── yelp.att.analysis
    │   ├── yelp.interaction.graph
    │   ├── yelp.social.graph
    │   ├── yelp.test.rating.712
    │   ├── yelp.train.rating.712
    │   └── yelp.val.rating.712
    ├── yelp.ipynb


├── log  # saved log file
│   ├── wxt.0.0.6023.0.35225.model
```



## Dataset 

FWD dataset (i.e., wxt data) can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10R8_ESb4fYW0WLPIBVn3g8si3mNIeqY1?usp=sharing) and [BaiduYun](https://pan.baidu.com/s/1zNagRTvdOwsONAFBtM8IFg) （提取码：i6qy）



## Reproducing results in the paper

Load the saved models in `log/` dir.



## Training

```
python trainHeteInf.py  --help
```



## Reference

```
@inproceedings{Yuanfu2020SIAN,
  title={Social Influence Attentive Neural Network for Friend-Enhanced Recommendation},
  author={Yuanfu Lu, Ruobing Xie, Chuan Shi, Yuan Fang, Wei Zhang, Xu Zhang, Leyu Lin.}
  booktitle={Proceedings of ECML-PKDD},
  year={2020}
}
```
