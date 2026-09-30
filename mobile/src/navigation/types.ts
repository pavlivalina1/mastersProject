import { ClassificationResult } from '../ml/useCropClassifier';

export type HomeStackParamList = {
  Home: undefined;
  Capture: { source: 'camera' | 'gallery' };
  DemoSamples: undefined;
  Result: { imageUri: string; result: ClassificationResult };
};

export type RootTabParamList = {
  HomeTab: undefined;
  HistoryTab: undefined;
};
