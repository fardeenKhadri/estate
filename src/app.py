import gradio as gr
import numpy as np
import os
import torch
from PIL import Image
from utils.logger import get_logger
from config.defaults import get_config
from inference import preprocess, run_one_inference
from models.build import build_model
from argparse import Namespace
from pano.create_panorama import create_panorama  # Import panorama stitching function
import gdown

# Logger setup
logger = get_logger()

# Function to download model checkpoints
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

# Main pipeline function
def process_pipeline(image_files, pre_processing, weight_name, post_processing, visualization, mesh_format, mesh_resolution):
    # Step 1: Stitch images into a panorama
    output_panorama = "src/output/panorama.jpg"
    panorama = create_panorama(image_files, output_panorama)

    if panorama is None:
        return "Error: Panorama creation failed. Ensure at least two valid images are uploaded.", None, None, None, None

    # Step 2: Select the model based on user choice
    args.pre_processing = pre_processing
    args.post_processing = post_processing

    if weight_name == 'mp3d':
        model = mp3d_model
    elif weight_name == 'zind':
        model = zind_model
    else:
        logger.error("Unknown pre-trained weight name")
        raise NotImplementedError

    # Step 3: Preprocess and run inference on the panorama
    img_name = os.path.basename(output_panorama).split('.')[0]
    img = np.array(Image.open(output_panorama).resize((1024, 512), Image.Resampling.BICUBIC))[..., :3]

    vp_cache_path = 'src/demo/default_vp.txt'
    if args.pre_processing:
        vp_cache_path = os.path.join('src/output', f'{img_name}_vp.txt')
        logger.info("Pre-processing...")
        img, vp = preprocess(img, vp_cache_path=vp_cache_path)

    img = (img / 255.0).astype(np.float32)
    run_one_inference(img, model, args, img_name,
                      logger=logger, show=False,
                      show_depth='depth-normal-gradient' in visualization,
                      show_floorplan='2d-floorplan' in visualization,
                      mesh_format=mesh_format, mesh_resolution=int(mesh_resolution))

    return [os.path.join(args.output_dir, f"{img_name}_pred.png"),
            os.path.join(args.output_dir, f"{img_name}_3d{mesh_format}"),
            os.path.join(args.output_dir, f"{img_name}_3d{mesh_format}"),
            vp_cache_path,
            os.path.join(args.output_dir, f"{img_name}_pred.json")]

# Function to load model
def get_model(args):
    config = get_config(args)
    down_ckpt(args.cfg, config.CKPT.DIR)
    if ('cuda' in args.device or 'cuda' in config.TRAIN.DEVICE) and not torch.cuda.is_available():
        logger.info(f"The {args.device} is not available, switching to CPU...")
        config.defrost()
        args.device = "cpu"
        config.TRAIN.DEVICE = "cpu"
        config.freeze()
    model, _, _, _ = build_model(config, logger)
    return model

# Main execution
if __name__ == '__main__':
    args = Namespace(device='cuda', output_dir='src/output', visualize_3d=False, output_3d=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    args.cfg = 'src/config/mp3d.yaml'
    mp3d_model = get_model(args)

    args.cfg = 'src/config/zind.yaml'
    zind_model = get_model(args)

    # Gradio interface
    description = "This demo allows you to create a 3D room layout using stitched panorama images. " \
                  "Upload multiple images for stitching and select model settings for inference."

    demo = gr.Interface(
        fn=process_pipeline,
        inputs=[
            gr.File(label="Input Images", type="file", file_types=[".jpg", ".png"], multiple=True),
            gr.Checkbox(label="Pre-processing", value=True),
            gr.Radio(['mp3d', 'zind'], label="Pre-trained Weight", value='mp3d'),
            gr.Radio(['manhattan', 'atalanta', 'original'], label="Post-processing Method", value='manhattan'),
            gr.CheckboxGroup(['depth-normal-gradient', '2d-floorplan'], label="2D Visualization",
                             value=['depth-normal-gradient', '2d-floorplan']),
            gr.Radio(['.gltf', '.obj', '.glb'], label="3D Mesh Format", value='.gltf'),
            gr.Radio(['128', '256', '512', '1024'], label="3D Mesh Resolution", value='256')
        ],
        outputs=[
            gr.Image(label="Predicted Result 2D Visualization"),
            gr.Model3D(label="3D Mesh Reconstruction"),
            gr.File(label="3D Mesh File"),
            gr.File(label="Vanishing Point Information"),
            gr.File(label="Layout JSON")
        ],
        title="3D Room Layout Estimation with Panorama Stitching",
        description=description,
        allow_flagging="never",
        cache_examples=False
    )

    # Launch the app on port 6106
    demo.launch(debug=True, enable_queue=False, server_port=6106)
