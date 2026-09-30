// Must mirror src/config.py exactly (IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD)
// and the input/output names used in src/export_mobile.py's torch.onnx.export.
export const IMAGE_SIZE = 128;
export const IMAGENET_MEAN = [0.485, 0.456, 0.406];
export const IMAGENET_STD = [0.229, 0.224, 0.225];

export const MODEL_INPUT_NAME = 'input';
export const MODEL_LOGITS_OUTPUT_NAME = 'logits';
export const MODEL_CAM_OUTPUT_NAME = 'cam';
export const MODEL_ANDROID_ASSET_PATH = 'models/mobilenet_v2_da.onnx';
export const MODEL_FILE_NAME = 'mobilenet_v2_da.onnx';

// Spatial size of the CAM output — mobilenet_v2's last conv feature map at
// 128x128 input (see src/export_mobile.py's _WithCAM).
export const CAM_GRID_SIZE = 4;

export const CONFIDENCE_THRESHOLD = 0.6;
