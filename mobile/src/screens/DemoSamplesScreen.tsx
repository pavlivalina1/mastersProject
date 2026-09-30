import React, { useState } from 'react';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { ActivityIndicator, FlatList, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { Card } from '../components/Card';
import { colors } from '../theme/colors';
import { typography } from '../theme/typography';
import { HomeStackParamList } from '../navigation/types';
import { DEMO_SAMPLES } from '../demoSamples';
import { useCropClassifier } from '../ml/useCropClassifier';

type Props = NativeStackScreenProps<HomeStackParamList, 'DemoSamples'>;

// Lets you demo/test the classifier with bundled sample photos instead of a
// real camera/gallery photo — handy for presentations or quick sanity checks.
export function DemoSamplesScreen({ navigation }: Props) {
  const { classify } = useCropClassifier();
  const [busyKey, setBusyKey] = useState<string | null>(null);

  const onPick = async (key: string, uri: string) => {
    if (busyKey) return;
    setBusyKey(key);
    try {
      const result = await classify(uri);
      navigation.navigate('Result', { imageUri: uri, result });
    } finally {
      setBusyKey(null);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.intro}>
        Готові приклади фото для швидкої перевірки - без потреби фотографувати
        реальний листок
      </Text>
      <FlatList
        data={DEMO_SAMPLES}
        keyExtractor={item => item.key}
        contentContainerStyle={styles.list}
        renderItem={({ item }) => (
          <Pressable onPress={() => onPick(item.key, item.uri)} disabled={busyKey !== null}>
            <Card style={styles.row}>
              <Image source={{ uri: item.uri }} style={styles.thumb} />
              <View style={styles.rowText}>
                <Text style={styles.label}>{item.label}</Text>
                <Text style={styles.hint}>{item.hint}</Text>
              </View>
              {busyKey === item.key ? (
                <ActivityIndicator color={colors.primary} />
              ) : (
                <Text style={styles.chevron}>›</Text>
              )}
            </Card>
          </Pressable>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background, padding: 20, gap: 14 },
  intro: { ...typography.body, color: colors.textMuted },
  list: { gap: 12 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  thumb: { width: 56, height: 56, borderRadius: 12, backgroundColor: colors.border },
  rowText: { flex: 1 },
  label: { ...typography.subheading, fontSize: 15, color: colors.text },
  hint: { ...typography.caption, color: colors.textMuted, marginTop: 2 },
  chevron: { ...typography.heading, fontSize: 22, color: colors.textMuted },
});
