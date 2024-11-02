import gradio as gr
from PIL import Image
import os

import spaces

from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1"
)

@spaces.GPU(duration=160)
def generate_image(text, img1, img2, img3, height, width, guidance_scale, img_guidance_scale, inference_steps, seed, separate_cfg_infer, offload_model,
            use_input_image_size_as_output, max_input_image_size):
    input_images = [img1, img2, img3]
    # Delete None
    input_images = [img for img in input_images if img is not None]
    if len(input_images) == 0:
        input_images = None

    output = pipe(
        prompt=text,
        input_images=input_images,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        img_guidance_scale=img_guidance_scale,
        num_inference_steps=inference_steps,
        separate_cfg_infer=separate_cfg_infer, 
        use_kv_cache=True,
        offload_kv_cache=True,
        offload_model=offload_model,
        use_input_image_size_as_output=use_input_image_size_as_output,
        seed=seed,
        max_input_image_size=max_input_image_size,
    )
    img = output[0]
    return img



def get_example():
    case = [
        [
            "A curly-haired man in a red shirt is drinking tea.",
            None,
            None,
            None,
            1024,
            1024,
            2.5,
            1.6,
            50,
            0,
            True,
            False,
            False,
            1024,
        ],
        [
            "The woman in <img><|image_1|></img> waves her hand happily in the crowd",
            "./imgs/test_cases/zhang.png",
            None,
            None,
            1024,
            1024,
            2.5,
            1.9,
            50,
            128,
            True,
            False,
            False,
            1024,
        ],
        [
            "A man in a black shirt is reading a book. The man is the right man in <img><|image_1|></img>.",
            "./imgs/test_cases/two_man.jpg",
            None,
            None,
            1024,
            1024,
            2.5,
            1.6,
            50,
            0,
            True,
            False,
            False,
            1024,
        ],
        [
            "Two woman are raising fried chicken legs in a bar. A woman is <img><|image_1|></img>. The other woman is <img><|image_2|></img>.",
            "./imgs/test_cases/mckenna.jpg",
            "./imgs/test_cases/Amanda.jpg",
            None,
            1024,
            1024,
            2.5,
            1.8,
            50,
            168,
            True,
            False,
            False,
            1024,
        ],
        [
            "A man and a short-haired woman with a wrinkled face are standing in front of a bookshelf in a library. The man is the man in the middle of <img><|image_1|></img>, and the woman is oldest woman in <img><|image_2|></img>",
            "./imgs/test_cases/1.jpg",
            "./imgs/test_cases/2.jpg",
            None,
            1024,
            1024,
            2.5,
            1.6,
            50,
            60,
            True,
            False,
            False,
            1024,
        ],
        [
            "A man and a woman are sitting at a classroom desk. The man is the man with yellow hair in <img><|image_1|></img>. The woman is the woman on the left of <img><|image_2|></img>",
            "./imgs/test_cases/3.jpg",
            "./imgs/test_cases/4.jpg",
            None,
            1024,
            1024,
            2.5,
            1.8,
            50,
            66,
            True,
            False,
            False,
            1024,
        ],
        [
            "The flower <img><|image_1|><\/img> is placed in the vase which is in the middle of <img><|image_2|><\/img> on a wooden table of a living room",
            "./imgs/test_cases/rose.jpg",
            "./imgs/test_cases/vase.jpg",
            None,
            1024,
            1024,
            2.5,
            1.6,
            50,
            0,
            True,
            False,
            False,
            1024,
        ],
        [
            "<img><|image_1|><img>\n Remove the woman's earrings. Replace the mug with a clear glass filled with sparkling iced cola.",
            "./imgs/demo_cases/t2i_woman_with_book.png",
            None,
            None,
            None,
            None,
            2.5,
            1.6,
            50,
            222,
            True,
            False,
            True,
            1024,
        ],
        [
            "Detect the skeleton of human in this image: <img><|image_1|></img>.",
            "./imgs/test_cases/control.jpg",
            None,
            None,
            None,
            None,
            2.0,
            1.6,
            50,
            0,
            True,
            False,
            True,
            1024,
        ],
        [
            "Generate a new photo using the following picture and text as conditions: <img><|image_1|><img>\n A young boy is sitting on a sofa in the library, holding a book. His hair is neatly combed, and a faint smile plays on his lips, with a few freckles scattered across his cheeks. The library is quiet, with rows of shelves filled with books stretching out behind him.",
            "./imgs/demo_cases/skeletal.png",
            None,
            None,
            None,
            None,
            2,
            1.6,
            50,
            42,
            True,
            False,
            True,
            1024,
        ],
        [
            "Following the pose of this image <img><|image_1|><img>, generate a new photo: A young boy is sitting on a sofa in the library, holding a book. His hair is neatly combed, and a faint smile plays on his lips, with a few freckles scattered across his cheeks. The library is quiet, with rows of shelves filled with books stretching out behind him.",
            "./imgs/demo_cases/edit.png",
            None,
            None,
            None,
            None,
            2.0,
            1.6,
            50,
            123,
            True,
            False,
            True,
            1024,
        ],
        [
            "Following the depth mapping of this image <img><|image_1|><img>, generate a new photo: A young girl is sitting on a sofa in the library, holding a book. His hair is neatly combed, and a faint smile plays on his lips, with a few freckles scattered across his cheeks. The library is quiet, with rows of shelves filled with books stretching out behind him.",
            "./imgs/demo_cases/edit.png",
            None,
            None,
            None,
            None,
            2.0,
            1.6,
            50,
            1,
            True,
            False,
            True,
            1024,
        ],
        [
            "<img><|image_1|><\/img> What item can be used to see the current time? Please remove it.",
            "./imgs/test_cases/watch.jpg",
            None,
            None,
            None,
            None,
            2.5,
            1.6,
            50,
            0,
            True,
            False,
            True,
            1024,
        ],
        [
            "According to the following examples, generate an output for the input.\nInput: <img><|image_1|></img>\nOutput: <img><|image_2|></img>\n\nInput: <img><|image_3|></img>\nOutput: ",
            "./imgs/test_cases/icl1.jpg",
            "./imgs/test_cases/icl2.jpg",
            "./imgs/test_cases/icl3.jpg",
            224,
            224,
            2.5,
            1.6,
            50,
            1,
            True,
            False,
            False,
            768,
        ],
    ]
    return case

def run_for_examples(text, img1, img2, img3, height, width, guidance_scale, img_guidance_scale, inference_steps, seed, separate_cfg_infer, offload_model,
            use_input_image_size_as_output, max_input_image_size):    
    return generate_image(text, img1, img2, img3, height, width, guidance_scale, img_guidance_scale, inference_steps, seed, separate_cfg_infer, offload_model,
            use_input_image_size_as_output, max_input_image_size)

description = """
OmniGen is a unified image generation model that you can use to perform various tasks, including but not limited to text-to-image generation, subject-driven generation, Identity-Preserving Generation, and image-conditioned generation.
For multi-modal to image generation, you should pass a string as `prompt`, and a list of image paths as `input_images`. The placeholder in the prompt should be in the format of `<img><|image_*|></img>` (for the first image, the placeholder is <img><|image_1|></img>. for the second image, the the placeholder is <img><|image_2|></img>).
For example, use an image of a woman to generate a new image:
prompt = "A woman holds a bouquet of flowers and faces the camera. Thw woman is \<img\>\<|image_1|\>\</img\>."

Tips:
- For out of memory or time cost, you can set `offload_model=True` or refer to [./docs/inference.md#requiremented-resources](https://github.com/VectorSpaceLab/OmniGen/blob/main/docs/inference.md#requiremented-resources) to select a appropriate setting.
- If inference time is too long when input multiple images, please try to reduce the `max_input_image_size`. More details please refer to [./docs/inference.md#requiremented-resources](https://github.com/VectorSpaceLab/OmniGen/blob/main/docs/inference.md#requiremented-resources).
- Oversaturated: If the image appears oversaturated, please reduce the `guidance_scale`.
- Not match the prompt: If the image does not match the prompt, please try to increase the `guidance_scale`.
- Low-quality: More detailed prompt will lead to better results. 
- Animate Style: If the genereate images is in animate style, you can try to add `photo` to the prompt`.
- Edit generated image. If you generate a image by omnigen and then want to edit it, you cannot use the same seed to edit this image. For example, use seed=0 to generate image, and should use seed=1 to edit this image.
- For image editing tasks, we recommend placing the image before the editing instruction. For example, use `<img><|image_1|></img> remove suit`, rather than `remove suit <img><|image_1|></img>`. 
- For image editing task and controlnet task, we recommend to set the height and width of output image as the same as input image. For example, if you want to edit a 512x512 image, you should set the height and width of output image as 512x512. You also can set the `use_input_image_size_as_output` to automatically set the height and width of output image as the same as input image.


"""

article = """
---
**Citation** 
<br> 
If you find this repository useful, please consider giving a star ⭐ and citation
```
@article{xiao2024omnigen,
  title={Omnigen: Unified image generation},
  author={Xiao, Shitao and Wang, Yueze and Zhou, Junjie and Yuan, Huaying and Xing, Xingrun and Yan, Ruiran and Wang, Shuting and Huang, Tiejun and Liu, Zheng},
  journal={arXiv preprint arXiv:2409.11340},
  year={2024}
}
```
**Contact**
<br>
If you have any questions, please feel free to open an issue or directly reach us out via email.
"""


# Gradio
with gr.Blocks() as demo:
    gr.Markdown("# OmniGen: Unified Image Generation [论文](https://arxiv.org/abs/2409.11340) [代码](https://github.com/VectorSpaceLab/OmniGen)")
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            # 文本提示
            prompt_input = gr.Textbox(
                label="输入您的提示，使用 <img><|image_i|></img> 来表示第 i 个输入图像", placeholder="在这里输入您的提示..."
            )

            with gr.Row(equal_height=True):
                # 输入图像
                image_input_1 = gr.Image(label="<img><|image_1|></img>", type="filepath")
                image_input_2 = gr.Image(label="<img><|image_2|></img>", type="filepath")
                image_input_3 = gr.Image(label="<img><|image_3|></img>", type="filepath")

            # 滑块
            height_input = gr.Slider(
                label="高度", minimum=128, maximum=2048, value=1024, step=16
            )
            width_input = gr.Slider(
                label="宽度", minimum=128, maximum=2048, value=1024, step=16
            )

            guidance_scale_input = gr.Slider(
                label="指导尺度", minimum=1.0, maximum=5.0, value=2.5, step=0.1
            )

            img_guidance_scale_input = gr.Slider(
                label="图像指导尺度", minimum=1.0, maximum=2.0, value=1.6, step=0.1
            )

            num_inference_steps = gr.Slider(
                label="推理步骤数", minimum=1, maximum=100, value=50, step=1
            )

            seed_input = gr.Slider(
                label="种子", minimum=0, maximum=2147483647, value=42, step=1
            )

            max_input_image_size = gr.Slider(
                label="最大输入图像尺寸", minimum=128, maximum=2048, value=1024, step=16
            )

            separate_cfg_infer = gr.Checkbox(
                label="分离 CFG 推理", info="是否为不同的指导使用单独的推理过程。这将减少内存消耗。", value=True,
            )
            offload_model = gr.Checkbox(
                label="卸载模型", info="将模型卸载到 CPU，这将显著减少内存消耗但会减慢生成速度。您可以取消分离 CFG 推理并设置卸载模型=True。如果分离 CFG 推理和卸载模型都为 True，将进一步减少内存，但生成速度最慢。", value=False,
            )
            use_input_image_size_as_output = gr.Checkbox(
                label="将输入图像尺寸用作输出", info="自动将输出图像尺寸调整为与输入图像尺寸相同。对于编辑和 ControlNet 任务，它可以确保输出图像与输入图像尺寸相同，从而实现更好的性能。", value=False,
            )

            # 生成按钮
            generate_button = gr.Button("生成图像")


        with gr.Column():
            # 输出图像
            output_image = gr.Image(label="输出图像")

    # 点击事件
    generate_button.click(
        generate_image,
        inputs=[
            prompt_input,
            image_input_1,
            image_input_2,
            image_input_3,
            height_input,
            width_input,
            guidance_scale_input,
            img_guidance_scale_input,
            num_inference_steps,
            seed_input,
            separate_cfg_infer,
            offload_model,
            use_input_image_size_as_output,
            max_input_image_size,
        ],
        outputs=output_image,
    )

    gr.Examples(
        examples=get_example(),
        fn=run_for_examples,
        inputs=[
            prompt_input,
            image_input_1,
            image_input_2,
            image_input_3,
            height_input,
            width_input,
            guidance_scale_input,
            img_guidance_scale_input,
            num_inference_steps,
            seed_input,
            separate_cfg_infer,
            offload_model,
            use_input_image_size_as_output,
            max_input_image_size,
        ],
        outputs=output_image,
    )

    gr.Markdown(article)

# 启动 Gradio 应用，启用分享功能
demo.launch(
    share=True,
    server_name="0.0.0.0",  # 绑定到所有网络接口
    server_port=7860,       # 选择一个端口（可选）
)
