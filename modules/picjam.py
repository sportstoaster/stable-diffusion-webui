import os
import shutil
import gradio as gr
import torch
# from dreambooth import dreambooth_row


def gpu_memory_cleanup():
    torch.cuda.empty_cache()

def style_change(style):
  if style=="Studio":
    return {env: gr.update(visible=False), bg: gr.update(visible=True)}
  else:
    return {env: gr.update(visible=True), bg: gr.update(visible=False)}

def images_upload(images, dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)
    
    for i, image in enumerate(images):
        image.save(os.path.join(dirname, f"image{i}.jpg"))

    return os.path.abspath(dirname)

def dreambooth_prompt(obj_name, obj_type):
    return f"photo of {obj_name} {obj_type}"

def preprocessing(images):
    return images


def predict(prompt, style, angle, back, num):
  # image =  dict['image'].convert("RGB").resize((512, 512))
  # prompt = "photo of OBJCTTST"
  # prompt = " photo of bag "

  angle_prompt = ""
  angle_prompt = angle + angle_prompt

  style_prompt = ""
  if style == "Studio":
    style_prompt=" in clean white studio environment"
  elif style=="Professional Lifestyle":
    style_prompt=" street, product photo, unit, office, high-fashion, DSLR"
  elif style=="UGC Lifestyle":
    pass
  
  bg_prompt = f" with {back} background"

  prompt += style_prompt + bg_prompt
  prompt = angle_prompt + " " + prompt
  print(prompt)
  # images = pipe(prompt=prompt, num_inference_steps=30, num_images_per_prompt=num).images
  images = []
  return (images, prompt)

def prompt_constructor(prompt, style, angle, background, bg_color, positiion, items):
    negative_prompt = ""
    angle_prompt = ""
    pos_prompt = ""
    bg_prompt = ""
    items_prompt = ""
    style_prompt = ""
    angle_prompt = angle + angle_prompt
    
    style_prompt = ""
    if style == "Studio":
        style_prompt=" in clean studio environment"
        negative_prompt += "street, product photo, unit, office, high-fashion, DSLR"
    elif style=="Professional Lifestyle":
        style_prompt=" street, product photo, unit, office, high-fashion, DSLR"
        negative_prompt += "in clean white studio environment"
    elif style=="UGC Lifestyle":
        style_prompt="iphone photo, instagram, review"
        negative_prompt += "in clean white studio environment, DSLR, camera, 85mm, 100mm"
    
    if len(background) > 0:
        bg_prompt = f" with {background} in background"
    if len(bg_color) > 0:
        bg_prompt = f" with {bg_color} background"
    if len(positiion) > 0:
        pos_prompt = f"on {positiion}"
    if len(items) > 0:
        items_prompt = f" with {items}"
    
    prompt += pos_prompt + items_prompt
    prompt += style_prompt + bg_prompt
    
    prompt = angle_prompt + " " + prompt

    return prompt, negative_prompt

import string, random

def dreambooth_train_params(images, obj_name=''):
    if len(obj_name) <= 1:
        # Generate random string
        obj_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
   
    if not os.path.exists(f'data/dbtrained/'):
        os.makedirs(f'data/dbtrained/')
    if not os.path.exists(f'data/dbtrained/{obj_name}'):
        os.makedirs(f'data/dbtrained/{obj_name}')
    
    dirname = images_upload(images, f'data/dbtrained/{obj_name}')
    return dirname, obj_name

def generate_wrapper(func, txt2img_args, style_args):

    prompt = txt2img_args['inputs'][0]
    prompt, negative_prompt = prompt_constructor(prompt, style_args['style'], style_args['angle'], style_args['background'], style_args['bg_color'], style_args['position'], style_args['items'])
    
    txt2img_args['inputs'][0] = prompt
    txt2img_args['inputs'][1] = negative_prompt

    return func(**txt2img_args)