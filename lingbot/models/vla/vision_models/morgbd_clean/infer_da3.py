import torch
from depth_anything_3.api import DepthAnything3

# Load model from Hugging Face Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/da3metric-large")
# model = DepthAnything3.from_pretrained(pretrained_model_name_or_path="/nas3/tanbin/pretrain_model_weights_zoom/da3/DA3METRIC-LARGE/DA3METRIC-LARGE")
model = model.to(device=device)

# Run inference on images
images = ["/nas3/tanbin/Depth-Anything-3/assets/examples/SOH/000.png", "/nas3/tanbin/Depth-Anything-3/assets/examples/SOH/000.png"]  # List of image paths, PIL Images, or numpy arrays
prediction = model.inference(
    images,
)
# Access results
print(prediction.depth.shape)        # Depth maps: [N, H, W] float32

import pdb; pdb.set_trace()