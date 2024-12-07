import os
import subprocess
import sys

def run_ppstructure_inference(
    image_dir,
    det_model_dir,
    rec_model_dir,
    rec_char_dict_path,
    table_model_dir,
    table_char_dict_path,
    layout_model_dir,
    layout_dict_path,
    vis_font_path,
    output_dir,
    recovery=True,
    return_word_box=True
):
    """
    Run PaddleOCR PPStructure inference with flexible path handling.
    
    Args:
        image_dir (str): Path to the input image directory
        det_model_dir (str): Path to detection model directory
        rec_model_dir (str): Path to recognition model directory
        rec_char_dict_path (str): Path to recognition character dictionary
        table_model_dir (str): Path to table model directory
        table_char_dict_path (str): Path to table character dictionary
        layout_model_dir (str): Path to layout model directory
        layout_dict_path (str): Path to layout dictionary
        vis_font_path (str): Path to visualization font
        output_dir (str): Path to output directory
        recovery (bool, optional): Enable recovery mode. Defaults to True.
        return_word_box (bool, optional): Return word bounding boxes. Defaults to True.
    
    Returns:
        subprocess.CompletedProcess: Result of the inference command
    """
    # Ensure all paths are absolute and normalized
    base_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    
    # Normalize all paths
    image_dir = os.path.abspath(image_dir)
    det_model_dir = os.path.abspath(det_model_dir)
    rec_model_dir = os.path.abspath(rec_model_dir)
    rec_char_dict_path = os.path.abspath(rec_char_dict_path)
    table_model_dir = os.path.abspath(table_model_dir)
    table_char_dict_path = os.path.abspath(table_char_dict_path)
    layout_model_dir = os.path.abspath(layout_model_dir)
    layout_dict_path = os.path.abspath(layout_dict_path)
    vis_font_path = os.path.abspath(vis_font_path)
    output_dir = os.path.abspath(output_dir)

    # Construct the command with platform-independent paths
    command = [
        sys.executable,  # Use the current Python interpreter
        'predict_system.py',
        f'--image_dir={image_dir}',
        f'--det_model_dir={det_model_dir}',
        f'--rec_model_dir={rec_model_dir}',
        f'--rec_char_dict_path={rec_char_dict_path}',
        f'--table_model_dir={table_model_dir}',
        f'--table_char_dict_path={table_char_dict_path}',
        f'--layout_model_dir={layout_model_dir}',
        f'--layout_dict_path={layout_dict_path}',
        f'--vis_font_path={vis_font_path}',
        f'--output={output_dir}',
        f'--recovery={str(recovery).lower()}',
        f'--return_word_box={str(return_word_box).lower()}'
    ]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Run the command from the ppstructure directory
        result = subprocess.run(
            command, 
            cwd=os.path.join(base_dir, 'ppstructure'),  # Ensure running from ppstructure directory
            check=True,  # Raise CalledProcessError if command returns non-zero exit code
            capture_output=True,  # Capture stdout and stderr
            text=True  # Return output as strings instead of bytes
        )
        
        print("Inference completed successfully.")
        return result
    
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")
        print(f"Standard Output: {e.stdout}")
        print(f"Standard Error: {e.stderr}")
        raise

def main():
    """
    Example usage of the inference function.
    Replace paths with your actual model and image paths.
    """
    # Get the base directory of the current script
    base_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    
    # Example paths (adjust these to match your project structure)
    image_path = os.path.join(base_dir, 'docs', 'ppstructure', 'images', 'table_1.png')
    det_model_dir = os.path.join(base_dir, 'inference', 'en_PP-OCRv3_det_infer')
    rec_model_dir = os.path.join(base_dir, 'inference', 'en_PP-OCRv3_rec_infer')
    rec_char_dict_path = os.path.join(base_dir, '..', 'ppocr', 'utils', 'en_dict.txt')
    table_model_dir = os.path.join(base_dir, 'inference', 'ppstructure_mobile_v2.0_SLANet_infer')
    table_char_dict_path = os.path.join(base_dir, '..', 'ppocr', 'utils', 'dict', 'table_structure_dict.txt')
    layout_model_dir = os.path.join(base_dir, 'inference', 'picodet_lcnet_x1_0_fgd_layout_infer')
    layout_dict_path = os.path.join(base_dir, '..', 'ppocr', 'utils', 'dict', 'layout_dict', 'layout_publaynet_dict.txt')
    vis_font_path = os.path.join(base_dir, '..', 'doc', 'fonts', 'simfang.ttf')
    output_dir = os.path.join(base_dir, '..', 'output')

    # Run the inference
    run_ppstructure_inference(
        image_dir=image_path,
        det_model_dir=det_model_dir,
        rec_model_dir=rec_model_dir,
        rec_char_dict_path=rec_char_dict_path,
        table_model_dir=table_model_dir,
        table_char_dict_path=table_char_dict_path,
        layout_model_dir=layout_model_dir,
        layout_dict_path=layout_dict_path,
        vis_font_path=vis_font_path,
        output_dir=output_dir
    )

if __name__ == '__main__':
    main()