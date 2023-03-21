import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
from rembg.bg import remove,pytorch_remove
import numpy as np
import io
from PIL import Image
from torchvision import utils
import os
import torchvision.transforms as transforms
from miscc.utils import get_code
from stylegan2.model import Generator
from torch.autograd import Variable

def remove_bg(input_path,output_path):
    f = np.fromfile(input_path)
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    image = Image.new('RGB', size=img.size, color=(255, 255, 255))
    image.paste(img, (0, 0), mask=img)
    image.save(output_path)




if __name__=='__main__':
    # input = 'datasets/birds/images/016.Painted_Bunting/Painted_Bunting_0091_15198.jpg'
    input = 'output/output2.png'
    output = 'output/1.png'
    remove_bg(input,output)
