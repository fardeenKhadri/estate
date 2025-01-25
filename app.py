import gradio as gr
import numpy as np
import os
import torch
from pano.create_panorama import create_panorama
os.system('pip install --upgrade --no-cache-dir gdown')

from PIL import Image
from utils.logger import get_logger
from config.defaults import get_config
from inference import preprocess, run_one_inference
from models.build import build_model
from argparse import Namespace
import gdown

def down_ckpt(model_cfg, ckpt_dir):
    model_ids = [
        ['src/config/mp3d.yaml', '1o97oAmd-yEP5bQrM0eAWFPLq27FjUDbh'],
        ['src/config/zind.yaml', '1PzBj-dfDfH_vevgSkRe5kczW0GVl_43I'],
        ['src/config/pano.yaml', '1JoeqcPbm_XBPOi6O9GjjWi3_rtyPZS8m'],
        ['src/config/s2d3d.yaml', '1PfJzcxzUsbwwMal7yTkBClIFgn8IdEzI'],
        ['src/config/ablation_study/full.yaml', '1U16TxUkvZlRwJNaJnq9nAUap-BhCVIha']
    ]

    for model_id in model_ids:
        if model_id[0] != model_cfg:
            continue
        path = os.path.join(ckpt_dir, 'best.pkl')
        if not os.path.exists(path):
            logger.info(f"Downloading {model_id}")
            os.makedirs(ckpt_dir, exist_ok=True)
            gdown.download(f"https://drive.google.com/uc?id={model_id[1]}", path, False)

def greet(img_path, pre_processing, weight_name, post_processing, visualization, mesh_format, mesh_resolution):
    args.pre_processing = pre_processing
    args.post_processing = post_processing
    if weight_name == 'mp3d':
        model = mp3d_model
    elif weight_name == 'zind':
        model = zind_model
    else:
        logger.error("Unknown pre-trained weight name")
        raise NotImplementedError

    img_name = os.path.basename(img_path).split('.')[0]
    img = np.array(Image.open(img_path).resize((1024, 512), Image.Resampling.BICUBIC))[..., :3]

    vp_cache_path = 'src/demo/default_vp.txt'
    if args.pre_processing:
        vp_cache_path = os.path.join('src/output', f'{img_name}_vp.txt')
        logger.info("Pre-processing ...")
        img, vp = preprocess(img, vp_cache_path=vp_cache_path)

    img = (img / 255.0).astype(np.float32)
    run_one_inference(img, model, args, img_name,
                      logger=logger, show=False,
                      show_depth='depth-normal-gradient' in visualization,
                      show_floorplan='2d-floorplan' in visualization,
                      mesh_format=mesh_format, mesh_resolution=int(mesh_resolution))

    # Generate and store .gltf file in 'src/output'
    gltf_file_path = os.path.join(args.output_dir, f"{img_name}.gltf")
    if mesh_format == '.gltf':
        logger.info(f"Saving .gltf file to {gltf_file_path}")
        with open(gltf_file_path, 'w') as gltf_file:
            gltf_file.write("Generated .gltf content")

    return [os.path.join(args.output_dir, f"{img_name}_pred.png"),
            os.path.join(args.output_dir, f"{img_name}_3d{mesh_format}"),
            gltf_file_path,  # Include .gltf file path in the outputs
            vp_cache_path,
            os.path.join(args.output_dir, f"{img_name}_pred.json")]

def get_model(args):
    config = get_config(args)
    down_ckpt(args.cfg, config.CKPT.DIR)
    if ('cuda' in args.device or 'cuda' in config.TRAIN.DEVICE) and not torch.cuda.is_available():
        logger.info(f"The {args.device} is not available, will use cpu...")
        config.defrost()
        args.device = "cpu"
        config.TRAIN.DEVICE = "cpu"
        config.freeze()
    model, _, _, _ = build_model(config, logger)
    return model

if __name__ == '__main__':
    logger = get_logger()
    args = Namespace(device='cuda', output_dir='src/output', visualize_3d=False, output_3d=True)
    os.makedirs(args.output_dir, exist_ok=True)

    args.cfg = 'src/config/mp3d.yaml'
    mp3d_model = get_model(args)

    args.cfg = 'src/config/zind.yaml'
    zind_model = get_model(args)

    description = "This demo of the github project " \
                  "<a href='https://github.com/zhigangjiang/LGT-Net' target='_blank'>LGT-Net</a>. <br/>If this project helped you, please add a star to the github project. " \
                  "<br/>It uses the Geometry-Aware Transformer Network to predict the 3d room layout of an rgb panorama."

    demo = gr.Interface(
        fn=greet,
        inputs=[
            gr.Image(type='filepath', label='Input RGB Panorama', value='F:\\AURIGO\\MARKVERSION 2\\src\\pano\\output_panorama.jpg'),
            gr.Checkbox(label='Pre-Processing', value=True),
            gr.Radio(['mp3d', 'zind'], label='Pre-Trained Weight', value='mp3d'),
            gr.Radio(['manhattan', 'atalanta', 'original'], label='Post-Processing Method', value='manhattan'),
            gr.CheckboxGroup(['depth-normal-gradient', '2d-floorplan'], label='2D Visualization', value=['depth-normal-gradient', '2d-floorplan']),
            gr.Radio(['.gltf', '.obj', '.glb'], label='Output Format of 3D Mesh', value='.gltf'),
            gr.Radio(['128', '256', '512', '1024'], label='Output Resolution of 3D Mesh', value='256'),
        ],
        outputs=[
            gr.Image(label='Predicted Result 2D-Visualization', type='filepath'),
            gr.Model3D(label='3D Mesh Reconstruction', clear_color=[1.0, 1.0, 1.0, 1.0]),
            gr.File(label='3D Mesh File'),
            gr.File(label='Vanishing Point Information'),
            gr.File(label='Layout JSON')
        ],
        examples=[
            ['F:\AURIGO\MARKVERSION 2\src\pano\output_panorama.jpg', True, 'mp3d', 'manhattan', ['depth-normal-gradient', '2d-floorplan'], '.gltf', '256'],
            # Add more examples as needed
        ],
        title='LGT-Net',
        allow_flagging="never",
        cache_examples=False,
        description=description
    )

    demo.launch(debug=True, enable_queue=False)
