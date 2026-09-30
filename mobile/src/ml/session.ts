import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
import { Platform } from 'react-native';
import {
  IMAGE_SIZE,
  MODEL_ANDROID_ASSET_PATH,
  MODEL_CAM_OUTPUT_NAME,
  MODEL_FILE_NAME,
  MODEL_INPUT_NAME,
  MODEL_LOGITS_OUTPUT_NAME,
} from './constants';

let sessionPromise: Promise<InferenceSession> | null = null;

export type InferenceOutput = {
  logits: Float32Array;
  cam: Float32Array; // [num_classes * CAM_GRID_SIZE * CAM_GRID_SIZE], row-major per class
};

// The native ONNX Runtime module needs a real filesystem path, not a Metro
// bundle reference, so the model is shipped as a raw Android asset
// (mobile/android/app/src/main/assets/models/...) and copied to the app's
// document directory on every cold start (always overwritten, not just
// copied-if-missing, so a model update in a new app build actually takes
// effect instead of being shadowed by a stale copy from a previous install).
async function ensureModelOnDisk(): Promise<string> {
  const destPath = `${RNFS.DocumentDirectoryPath}/${MODEL_FILE_NAME}`;

  if (Platform.OS === 'android') {
    if (await RNFS.exists(destPath)) {
      await RNFS.unlink(destPath);
    }
    await RNFS.copyFileAssets(MODEL_ANDROID_ASSET_PATH, destPath);
  } else {
    // iOS bundling isn't set up/verified — this repo's dev machine has no
    // macOS/Xcode to add the model as a bundle resource (see plan Context).
    throw new Error('iOS model bundling not implemented yet');
  }
  return destPath;
}

export async function getSession(): Promise<InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = ensureModelOnDisk().then(path => InferenceSession.create(path));
  }
  return sessionPromise;
}

export async function runInference(input: Float32Array): Promise<InferenceOutput> {
  const session = await getSession();
  const tensor = new Tensor('float32', input, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
  const results = await session.run({ [MODEL_INPUT_NAME]: tensor });
  return {
    logits: results[MODEL_LOGITS_OUTPUT_NAME].data as Float32Array,
    cam: results[MODEL_CAM_OUTPUT_NAME].data as Float32Array,
  };
}
