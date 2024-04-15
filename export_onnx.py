# git clone https://github.com/sczhou/CodeFormer
# cd CodeFormer
# pip install -r requirements.txt
# python basics/setup.py install
# then run this file inside that folder

import torch
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from basicsr.utils.registry import ARCH_REGISTRY

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}
device = get_device()
net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                        connect_list=['32', '64', '128', '256']).to(device)

ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                model_dir='weights/CodeFormer', progress=True, file_name=None)
checkpoint = torch.load(ckpt_path)['params_ema']
net.load_state_dict(checkpoint)
net.eval()

    
x = torch.rand(1, 3, 512, 512)
x = x.cuda()
    
torch.onnx.export(net,
                    x,
                    "./codeformer.onnx",
                    verbose=True,
                    input_names=['input'],
                    output_names=['output'],
                    opset_version=17,
                    export_params=True
                    )
