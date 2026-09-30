import React, { useEffect, useState } from 'react';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import RNFS from 'react-native-fs';
import { colors } from '../theme/colors';
import { typography } from '../theme/typography';
import { HomeStackParamList } from '../navigation/types';
import { pickImage } from '../utils/pickImage';
import { useCropClassifier } from '../ml/useCropClassifier';
import { addHistoryEntry } from '../store/historyStore';

type Props = NativeStackScreenProps<HomeStackParamList, 'Capture'>;

export function CaptureScreen({ route, navigation }: Props) {
  const { source } = route.params;
  const { classify } = useCropClassifier();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function run() {
      try {
        const pickedUri = await pickImage(source);
        if (!pickedUri) {
          navigation.goBack();
          return;
        }

        const result = await classify(pickedUri);
        if (cancelled) return;

        const scansDir = `${RNFS.DocumentDirectoryPath}/scans`;
        await RNFS.mkdir(scansDir);
        const savedPath = `${scansDir}/${Date.now()}.jpg`;
        await RNFS.copyFile(pickedUri.replace('file://', ''), savedPath);
        const savedUri = `file://${savedPath}`;

        addHistoryEntry({
          id: String(Date.now()),
          timestamp: Date.now(),
          imagePath: savedUri,
          speciesUk: result.top.speciesUk,
          conditionUk: result.top.conditionUk,
          healthy: result.top.healthy,
          confidence: result.top.confidence,
        });

        if (!cancelled) {
          navigation.replace('Result', { imageUri: savedUri, result });
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
        }
      }
    }

    run();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source]);

  return (
    <View style={styles.container}>
      {error ? (
        <>
          <Text style={styles.errorTitle}>Щось пішло не так</Text>
          <Text style={styles.errorBody}>{error}</Text>
        </>
      ) : (
        <>
          <ActivityIndicator size="large" color={colors.primary} />
          <Text style={styles.text}>Завантаження…</Text>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
    padding: 24,
  },
  text: { ...typography.body, color: colors.textMuted },
  errorTitle: { ...typography.subheading, color: colors.danger },
  errorBody: { ...typography.body, color: colors.textMuted, textAlign: 'center' },
});
