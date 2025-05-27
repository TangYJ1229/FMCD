# FMCD
## High-Resolution Remote Sensing Bitemporal Image Change Detection Based on Feature Interaction and Multitask Learning (TGRS 2022)

This repository is the official implementation:
> [High-Resolution Remote Sensing Bitemporal Image Change Detection Based on Feature Interaction and Multitask Learning](https://ieeexplore.ieee.org/document/10123082)  
> Chunhui Zhao, Yingjie Tang, Shou Feng, Yuanze Fan, Wei Li, Ran Tao, Lifu Zhang

## 📄 Abstract

With the development of remote sensing technology, high-resolution (HR) remote sensing optical images have gradually become the main source of change detection data. Albeit, the change detection for HR remote sensing images still faces challenges: 1) in complex scenes, a region contains a large amount of semantic information, which makes it difficult to accurately locate the boundaries between different semantics in the feature maps and 2) due to the inability to maintain consistent conditions such as light, weather, and other factors when acquiring bitemporal images, confounding factors such as the style of bitemporal data that are not related to change detection can cause detection difficulties. Therefore, a change detection method based on feature interaction and multitask learning (FMCD) is proposed in this article. To improve the ability to detect changes in complex scenes, FMCD models the context information of features through a multilevel feature interaction module, so as to obtain representative features, and to improve the sensitivity of the model to changes, the interaction between two temporal features is realized through the mix attention block (MAB). In addition, to eliminate the influence of weather and other factors, FMCD adopts a multitask learning strategy, takes domain adaptation as an auxiliary task, and maps the features of bitemporal images to the same space through the feature relationship adaptation module (FRAM) and feature distribution adaptation module (FDAM). Experiments on three datasets show that the proposed method is superior to other state-of-the-art methods.

## 🎮 Framework
![Framework](assets/framework.png)

## 📧 Contact

If you have any issues while using the project, please feel free to contact me: [tangyj@hrbeu.edu.cn](tangyj@hrbeu.edu.cn).

## 📚 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{zhao2023high,
  title={High-resolution remote sensing bitemporal image change detection based on feature interaction and multitask learning},
  author={Zhao, Chunhui and Tang, Yingjie and Feng, Shou and Fan, Yuanze and Li, Wei and Tao, Ran and Zhang, Lifu},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={61},
  pages={1--14},
  year={2023},
  publisher={IEEE}
}

```

## 📜 License

Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.
