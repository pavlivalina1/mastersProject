import { Skia, AlphaType, ColorType } from '@shopify/react-native-skia';
import { IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD } from './constants';

// Resizes the photo at `uri` to IMAGE_SIZE x IMAGE_SIZE (plain stretch, matching
// training's transforms.Resize((128,128)) — not a center-crop) and returns a
// normalized NCHW Float32Array [1,3,128,128], exactly mirroring
// src/data/dataset.py's base_transform(train=False) (Resize -> ToTensor ->
// Normalize(IMAGENET_MEAN, IMAGENET_STD)).
export async function imageUriToTensor(uri: string): Promise<Float32Array> {
  const data = await Skia.Data.fromURI(uri);
  if (!data) {
    throw new Error(`Failed to load image data from ${uri}`);
  }
  const image = Skia.Image.MakeImageFromEncoded(data);
  if (!image) {
    throw new Error(`Failed to decode image at ${uri}`);
  }

  const surface = Skia.Surface.MakeOffscreen(IMAGE_SIZE, IMAGE_SIZE);
  if (!surface) {
    throw new Error('Failed to create offscreen Skia surface');
  }
  const canvas = surface.getCanvas();
  const paint = Skia.Paint();
  canvas.drawImageRect(
    image,
    { x: 0, y: 0, width: image.width(), height: image.height() },
    { x: 0, y: 0, width: IMAGE_SIZE, height: IMAGE_SIZE },
    paint,
  );
  surface.flush();

  const pixels = surface.makeImageSnapshot().readPixels(0, 0, {
    width: IMAGE_SIZE,
    height: IMAGE_SIZE,
    colorType: ColorType.RGBA_8888,
    alphaType: AlphaType.Unpremul,
  }) as Uint8Array | null;
  if (!pixels) {
    throw new Error('Failed to read pixels from resized image');
  }

  return rgbaToNormalizedNchw(pixels);
}

function rgbaToNormalizedNchw(rgba: Uint8Array): Float32Array {
  const planeSize = IMAGE_SIZE * IMAGE_SIZE;
  const out = new Float32Array(3 * planeSize);
  for (let i = 0; i < planeSize; i++) {
    const r = rgba[i * 4] / 255;
    const g = rgba[i * 4 + 1] / 255;
    const b = rgba[i * 4 + 2] / 255;
    out[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    out[planeSize + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    out[2 * planeSize + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }
  return out;
}
