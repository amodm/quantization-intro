from PIL import Image
from vit.classnames import CLASS_NAMES
from vit.LRP import LRP
from vit.ViT import vit_base_patch16_224 as vit_LRP, VisionTransformer
import cv2
import numpy as np
import os
import streamlit as st
import torch
import torchvision.transforms as T

## hot patch cuda() calls to gpu() calls - so that this works on Apple Silicon as well
DEFAULT_BACKEND = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def to_gpu(self):
    if DEFAULT_BACKEND == 'cuda':
        return self.cuda()
    elif DEFAULT_BACKEND == 'mps':
        return self.to('mps')
    else:
        return self.to('cpu')

setattr(torch.Tensor, 'cuda', to_gpu)
setattr(torch.Tensor, 'gpu', to_gpu)
setattr(VisionTransformer, 'gpu', to_gpu)
## end hot patch


@st.cache_data
def open_image(path):
    return Image.open(path).resize((224, 224))


@st.cache_data
def get_sample_image_list():
    return sorted(os.listdir('samples'))


def get_attention_heatmap(image, attention):
    heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image)
    return cam / np.max(cam)


def f32_mask_from_sbits(sbits):
    sign_exp_mask = 0xff800000 # 1-bit of sign, 8-bits of exponent
    mantissa_mask = ((0x7fffff << (23 - sbits))) & 0x7fffff
    return sign_exp_mask | mantissa_mask


def reduce_precision_scalar(f32, sbits):
    mask = f32_mask_from_sbits(sbits)
    return np.uint32(np.float32(f32).view(np.uint32) & mask).view(np.float32)


def reduce_precision(model, sbits):
    mask = f32_mask_from_sbits(sbits)
    with torch.no_grad():
        for param in model.parameters():
            tensor = param.data
            arr1 = np.asarray(tensor.numpy(force = True), dtype=np.float32).view(np.uint32)
            arr2 = np.full(arr1.shape, mask, dtype=np.uint32)
            arr3 = np.bitwise_and(arr1, arr2).view(np.float32)
            param.data = torch.from_numpy(arr3).to(device = tensor.device)


def load_model():
    model = vit_LRP(pretrained=True).gpu()
    model.eval()
    return model


def generate_visualization(model, original_image, method='transformer_attribution'):
    attribution_generator = LRP(model)
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).gpu(), method=method).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = get_attention_heatmap(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def get_prediction_text(predictions, text = None):
    prob = torch.softmax(predictions, dim=1)
    output = f'{text} | ' if text else ''
    for idx in predictions.data.topk(3, dim=1)[1][0].tolist():
        cls = CLASS_NAMES[idx]
        output += f'{cls}={100 * prob[0][idx]:.2f}% | '
    return output


def main():
    st.write('# Effect of Precision on Transformer Attention')
    with st.container():
        test_image_file = st.selectbox('Choose a sample image', get_sample_image_list())
        test_image_file = 'samples/' + test_image_file
        uploaded_file = st.file_uploader('Or upload a file')
        if uploaded_file is not None:
            test_image_file = uploaded_file
        prop_method = st.selectbox('Propagation Method', ['transformer_attribution', 'rollout', 'last_layer'])
    
    if test_image_file is None:
        test_image_file = 'samples/cat.jpg'

    image = open_image(test_image_file)
    img_t = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])(image)

    with st.spinner('Loading model...'):
        model = load_model()

    with st.container():
        st.write('## Original')
        st.image(image)

    with st.container():
        st.write('## Attention Heatmap')
        sbits = st.slider('fractional bits', min_value=0, max_value=23, value=23)

        col1, col2 = st.columns(2)

        with col1:
            with st.spinner('Generating heatmap...'):
                obj = generate_visualization(model, img_t, prop_method)
                predictions = model(img_t.unsqueeze(0).gpu())
            st.image(obj, caption=get_prediction_text(predictions, 'attention heatmap with float32'), use_column_width=False)

        with col2:
            with st.spinner('Generating heatmap for reduced precision...'):
                reduce_precision(model, sbits)
                obj = generate_visualization(model, img_t, prop_method)
                predictions = model(img_t.unsqueeze(0).gpu())
            st.image(obj, caption=get_prediction_text(predictions, 'attention heatmap at reduced precision'), use_column_width=False)

    with st.container():
        st.write('## Effect on floating point values')
        st.write('π: ', np.pi)
        st.write('π at specified precision: ', reduce_precision_scalar(np.pi, sbits))


if __name__ == '__main__':
    main()
