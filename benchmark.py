import torch
import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from torchvision.transforms.functional import normalize
import argparse
import timeit

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def build_engine_context(trt_model_path):
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    trt.init_libnvinfer_plugins(None, '')
    
    engine = runtime.deserialize_cuda_engine(open(trt_model_path, 'rb').read())
    assert engine
    
    context = engine.create_execution_context()
    assert context

    return engine, context

def alloc_buffers(engine, context):
    inputs = []
    outputs = []
    allocations = []

    # Allocate buffers
    for i in range(engine.num_bindings):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT 
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        shape = context.get_tensor_shape(name)

        if is_input and shape[0] < 0:
            assert engine.num_optimization_profiles > 0
            profile_shape = engine.get_profile_shape(0, name)
            assert len(profile_shape) == 3 # min, opt, max
            context.set_input_shape(i, profile_shape[2])
            shape = context.get_tensor_shape(name)

        if is_input:
            batch_size = shape[0]

        size = dtype.itemsize
        for s in shape:
            size *= s

        allocation = cuda.mem_alloc(size)
        host_allocation = None if is_input else np.zeros(shape, dtype)
        binding = {
            'index': i,
            'name': name,
            'dtype': dtype,
            'shape': list(shape),
            'allocation': allocation,
            'host_allocation': host_allocation
        }

        allocations.append(allocation)
        if is_input:
            inputs.append(binding)
        else:
            outputs.append(binding)

    return inputs, outputs, allocations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=str, default='./input.png', help='Input image path', required=True)
    parser.add_argument('--output', type=str, default="./output.png", help='Output image path')
    parser.add_argument('--engine', type=str, default=None, help='Tensorrt engine path', required=True)
    args = parser.parse_args()
    
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)

    import pycuda.autoprimaryctx as ctx
    engine, context = build_engine_context(args.engine)    
    inputs, outputs, allocations = alloc_buffers(engine, context)

    img_t = img2tensor(img / 255., bgr2rgb=True, float32=True)
    normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    input_data = np.ascontiguousarray(img_t.cpu().numpy(), dtype=np.float32)


    for i in range(50):

        start_time = timeit.default_timer()
        
        cuda.memcpy_htod(inputs[0]['allocation'], input_data)
        context.execute_v2(allocations)
        cuda.memcpy_dtoh(outputs[2]['host_allocation'], outputs[2]['allocation'])
        output = outputs[2]['host_allocation']
        output=torch.from_numpy(output)

        end_time = timeit.default_timer()
        print(f"Execution time: {end_time-start_time} seconds")

        restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
        restored_face = restored_face.astype('uint8')

        cv2.imwrite(args.output,restored_face)