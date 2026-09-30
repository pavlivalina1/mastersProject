import {
  launchCamera,
  launchImageLibrary,
  ImagePickerResponse,
} from 'react-native-image-picker';

// Returns the picked photo's URI, or null if the user cancelled.
export async function pickImage(source: 'camera' | 'gallery'): Promise<string | null> {
  const options = { mediaType: 'photo' as const, quality: 0.8 as const, saveToPhotos: false };

  const response: ImagePickerResponse =
    source === 'camera' ? await launchCamera(options) : await launchImageLibrary(options);

  if (response.didCancel) {
    return null;
  }
  if (response.errorCode) {
    throw new Error(response.errorMessage ?? `Image picker error: ${response.errorCode}`);
  }
  const uri = response.assets?.[0]?.uri;
  if (!uri) {
    throw new Error('No image returned from picker');
  }
  return uri;
}
