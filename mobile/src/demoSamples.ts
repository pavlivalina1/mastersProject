import { Image } from 'react-native';

export type DemoSample = {
  key: string;
  label: string;
  hint: string;
  uri: string;
};

const RAW_SAMPLES = [
  {
    key: 'apple_rust',
    label: 'Яблуня з іржею',
    hint: 'Впевнений діагноз',
    source: require('../assets/demo_samples/apple_rust.jpg'),
  },
  {
    key: 'tomato_healthy',
    label: 'Здоровий томат',
    hint: 'Здорова рослина',
    source: require('../assets/demo_samples/tomato_healthy.jpg'),
  },
  {
    key: 'potato_late_blight',
    label: 'Картопля з фітофторозом',
    hint: 'Впевнений діагноз',
    source: require('../assets/demo_samples/potato_late_blight.jpg'),
  },
  {
    key: 'corn_rust',
    label: 'Кукурудза з іржею',
    hint: 'Впевнений діагноз',
    source: require('../assets/demo_samples/corn_rust.jpg'),
  },
  {
    key: 'uncertain_example',
    label: 'Складний випадок',
    hint: 'Приклад "не впевнена" відповіді',
    source: require('../assets/demo_samples/uncertain_example.jpg'),
  },
] as const;

export const DEMO_SAMPLES: DemoSample[] = RAW_SAMPLES.map(s => ({
  key: s.key,
  label: s.label,
  hint: s.hint,
  uri: Image.resolveAssetSource(s.source)!.uri,
}));
