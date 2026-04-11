
<div align="center">

<h1>DINO-Tok: Adapting DINO for Visual Tokenizers</h1>

[![arXiv](https://img.shields.io/badge/ArXiv-2511.20565-%23840707.svg)](https://arxiv.org/abs/2511.20565v2) 
</div>



[Mingkai Jia](https://scholar.google.com/citations?user=fcpTdvcAAAAJ&hl=zh-CN)<sup>1,2</sup>, [Mingxiao Li](https://lmx760581375.github.io/)<sup>2</sup>, [Zhijian Shu](https://garlicba.github.io/)<sup>2,3</sup>, [Anlin Zheng](https://yexiguafuqihao.github.io/)<sup>4</sup>, [Liaoyuan Fan](https://github.com/MKJia/DINO-Tok)<sup>2</sup>, [Jiaxin Guo](https://wrld.github.io/)<sup>5</sup>, [Tianxing Shi](https://github.com/MKJia/DINO-Tok)<sup>3</sup>, [Dongyue Lu](https://dylanorange.github.io/)<sup>2</sup>, [Zeming Li](https://www.zemingli.com/)<sup>1</sup>, [Xiaoyang Guo](https://xy-guo.github.io/)<sup>2</sup>, [Xiaojuan Qi](https://xjqi.github.io/)<sup>4</sup>, [Xiao-Xiao Long](https://www.xxlong.site/)<sup>3</sup>, [Qian Zhang](https://scholar.google.com.hk/citations?hl=zh-CN&user=pCY-bikAAAAJ)<sup>2</sup>, [Ping Tan](https://scholar.google.com/citations?user=XhyKVFMAAAAJ&hl=en)<sup>1*</sup>, [Wei Yin](https://yvanyin.net/)<sup>2*§</sup>, 

[HKUST](https://hkust.edu.hk/)<sup>1</sup>, [Horizon Robotics](https://en.horizon.auto/)<sup>2</sup>, [NJU](https://www.nju.edu.cn/)<sup>3</sup>, [HKU](https://hku.hk/)<sup>4</sup>, [CUHK](https://cuhk.edu.hk/)<sup>5</sup>, <br>
<sup>*</sup> Corresponding Author, <sup>§</sup> Project Leader
<br><br><image src="./assets/teaser.png"/>
</div>


## 🚀News
- ```[April 2026]``` Released Inference Code
- ```[April 2026]``` Released [models & stats](https://huggingface.co/mkjia/DINO-Tok/tree/main).
- ```[Nov 2025]``` Released [paper](https://arxiv.org/abs/2511.20565v2).

## 🔨TO DO LIST
- [ ] Training code.
- [x] Models & Evaluation code.
- [x] Huggingface models & stats.

## 🔑 Quick Start
<a id="quick start"></a>

### Installation

```bash
git clone https://github.com/MKJia/DINO-Tok.git
cd DINO-Tok
```

### Prepare env
```bash
conda create -n dinotok python=3.10
conda activate dinotok
pip3 install -r requirements.txt
```

### Download models
Download the pretrained models & stats from our [model & stat](https://huggingface.co/mkjia/DINO-Tok/tree/main) to your `/path/to/your/ckpt`.

### Data Preparation
We default use the ImageNet-1k dataset. Or you can try our UHDBench dataset on [huggingface](https://huggingface.co/datasets/mkjia/UHDBench/tree/main) and download to your `/path/to/your/dataset`.

### Evaluation
Remember to change the paths in scripts.
```bash
bash scripts/test_aetok.bash
bash scripts/test_aegen.bash
bash scripts/test_vqtok.bash
bash scripts/test_vqgen.bash
```

## 🗄️Demos
- 🔥 Qualitative reconstruction images. 

<image src="./assets/qual_rec.png"/>

- 🔥 Qualitative class-to-image generation of Imagenet.

<image src="./assets/qual_gen.png"/>

- 🔥 Evaluation of dino-tok-ae on 256×256 ImageNet benchmark. 

<image src="./assets/tab_ae.png"/>

- 🔥 Evaluation of dino-tok-vq on 256×256 ImageNet benchmark. 

<image src="./assets/tab_vq.png"/>


## 📌 Citation

If the paper and code from `DINO-Tok` help your research, we kindly ask you to give a citation to our paper ❤️. Additionally, if you appreciate our work and find this repository useful, giving it a star ⭐️ would be a wonderful way to support our work. Thank you very much.

```bibtex
@article{jia2025dinotok,
  title={DINO-Tok: Adapting DINO for Visual Tokenizers},
  author={Jia, Mingkai and Li, Mingxiao and Fan, Liaoyuan and Shi, Tianxing and Guo, Jiaxin and Li, Zeming and Guo, Xiaoyang and Long, Xiao-Xiao and Zhang, Qian and Tan, Ping and others},
  journal={arXiv preprint arXiv:2511.20565},
  year={2025}
}
```

## License

This repository is under the MIT License. For more license questions, please contact Mingkai Jia (mjiaab@connect.ust.hk) and Wei Yin (yvanwy@outlook.com).
