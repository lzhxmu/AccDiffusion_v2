# AccDiffusion v2 <p align="center">
<a href="https://arxiv.org/abs/2412.02099"> <img src='https://img.shields.io/badge/arXiv-2412.02099-b31b1b.svg'></a> 
</p>

Code release for "AccDiffusion v2: Towards More Accurate
Higher-Resolution Diffusion Extrapolation".




## To run the code, please follow the steps below:
### Set up the dependencies as:
```
conda create -n AccDiffusion python=3.9
conda activate AccDiffusion
pip install -r requirements.txt
```

## Higher-image generation
```
python accdiffusion_plus.py --experiment_name="AccDiffusionv2" \
    --model_ckpt="stabilityai/stable-diffusion-xl-base-1.0" \ # your sdxl model ckpt path
    --prompt="a cat and a dog are playing on the lawn." \
    --num_inference_steps=50 \
    --seed=2 \
    --resolution="4096,4096" \
    --upscale_mode="bicubic_latent" \
    --stride=64 \
    --c=0.3 \ # c can be adjusted based on the degree of repetition and quality of the generated image
    --use_progressive_upscaling  --use_skip_residual --use_multidiffusion --use_dilated_sampling --use_guassian \
    --use_md_prompt    --shuffle   --use_controlnet  --controlnet_conditioning_scale 0.6
``` 
The hyperparameters c can be adjusted based on the degree of repetition and quality of the generated image. The quality of the high-resolution images generated is greatly related to the seed. 

## Affiliation
1. Shanghai Innovation Institute
2. Xiamen University
3. Rakuten


## Citation
If you find this code useful, please cite our paper:
```
@article{lin2025accdiffusion,
  title={AccDiffusion v2: Towards More Accurate Higher-Resolution Diffusion Extrapolation},
  author={Lin, Zhihang and Lin, Mingbao and Zhan, Wengyi and Ji, Rongrong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
```