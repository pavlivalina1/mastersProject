import React, { useRef, useState } from 'react';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { Alert, Image, LayoutChangeEvent, StyleSheet, Switch, Text, View } from 'react-native';
import { captureRef } from 'react-native-view-shot';
import Share from 'react-native-share';
import { Card } from '../components/Card';
import { Button } from '../components/Button';
import { HeatmapOverlay } from '../components/HeatmapOverlay';
import { colors } from '../theme/colors';
import { typography } from '../theme/typography';
import { HomeStackParamList } from '../navigation/types';

type Props = NativeStackScreenProps<HomeStackParamList, 'Result'>;

type VisualState = 'healthy' | 'diseased' | 'uncertain';

const PHOTO_HEIGHT = 220;

export function ResultScreen({ route }: Props) {
  const { imageUri, result } = route.params;
  const { top, top3, isConfident, camGrid } = result;

  const [showHeatmap, setShowHeatmap] = useState(false);
  const [photoWidth, setPhotoWidth] = useState(0);
  const [sharing, setSharing] = useState(false);
  const shareTargetRef = useRef<React.ComponentRef<typeof View>>(null);

  const state: VisualState = !isConfident ? 'uncertain' : top.healthy ? 'healthy' : 'diseased';
  const accentColor =
    state === 'healthy' ? colors.secondary : state === 'diseased' ? colors.danger : colors.accent;

  const onPhotoLayout = (e: LayoutChangeEvent) => setPhotoWidth(e.nativeEvent.layout.width);

  const shareText =
    state === 'uncertain'
      ? `CropDoctor: не вдалося впевнено визначити хворобу (найближче — ${top.speciesUk}, ${top.conditionUk}).`
      : state === 'healthy'
      ? `CropDoctor: рослина (${top.speciesUk}) здорова!`
      : `CropDoctor: ${top.speciesUk} — ${top.conditionUk} (впевненість ${Math.round(top.confidence * 100)}%).\n${top.careTipUk}`;

  const onShare = async () => {
    if (sharing) return;
    setSharing(true);
    try {
      const uri = await captureRef(shareTargetRef, { format: 'png', quality: 0.9 });
      await Share.open({ url: `file://${uri}`, type: 'image/png', message: shareText, failOnCancel: false });
    } catch (e) {
      Alert.alert('Не вдалося поділитися', e instanceof Error ? e.message : String(e));
    } finally {
      setSharing(false);
    }
  };

  return (
    <View style={styles.container}>
      <View ref={shareTargetRef} collapsable={false} style={styles.shareTarget}>
        <View style={styles.photoContainer} onLayout={onPhotoLayout}>
          {imageUri ? <Image source={{ uri: imageUri }} style={styles.photo} /> : null}
          {showHeatmap && photoWidth > 0 && (
            <HeatmapOverlay grid={camGrid} width={photoWidth} height={PHOTO_HEIGHT} />
          )}
        </View>

        <Card style={[styles.card, { borderColor: accentColor }]}>
          {state === 'uncertain' ? (
            <>
              <Text style={styles.emoji}>🔍</Text>
              <Text style={[styles.title, { color: accentColor }]}>
                Не вдалося впевнено визначити
              </Text>
              <Text style={styles.body}>
                Спробуйте зняти фото ще раз: більше світла, один листок у кадрі, без
                сторонніх предметів на фоні.
              </Text>
              <Text style={styles.candidatesLabel}>Можливі варіанти:</Text>
              <View style={styles.chipRow}>
                {top3.map(c => (
                  <View key={c.labelId} style={styles.chip}>
                    <Text style={styles.chipText}>
                      {c.speciesUk} · {c.healthy ? 'здорова' : c.conditionUk} (
                      {Math.round(c.confidence * 100)}%)
                    </Text>
                  </View>
                ))}
              </View>
            </>
          ) : (
            <>
              <Text style={styles.emoji}>{state === 'healthy' ? '✅' : '🍃'}</Text>
              <Text style={[styles.title, { color: accentColor }]}>
                {state === 'healthy' ? 'Рослина здорова' : top.conditionUk}
              </Text>
              <Text style={styles.species}>{top.speciesUk}</Text>
              <Text style={styles.confidence}>
                Впевненість: {Math.round(top.confidence * 100)}%
              </Text>
              {!top.healthy && <Text style={styles.careTip}>{top.careTipUk}</Text>}
            </>
          )}
        </Card>
      </View>

      <View style={styles.toggleRow}>
        <Text style={styles.toggleLabel}>Показати теплокарту (на що дивилась модель)</Text>
        <Switch
          value={showHeatmap}
          onValueChange={setShowHeatmap}
          trackColor={{ true: colors.secondary, false: colors.border }}
        />
      </View>

      <Button
        label={sharing ? 'Готуємо…' : 'Поділитися результатом'}
        variant="secondary"
        onPress={onShare}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background, padding: 20, gap: 14 },
  shareTarget: { backgroundColor: colors.background, gap: 16 },
  photoContainer: { width: '100%', height: PHOTO_HEIGHT, borderRadius: 20, overflow: 'hidden' },
  photo: { width: '100%', height: '100%', backgroundColor: colors.border },
  card: { gap: 6, alignItems: 'center', borderWidth: 1 },
  emoji: { fontSize: 40 },
  title: { ...typography.subheading, fontSize: 20, textAlign: 'center' },
  species: { ...typography.body, color: colors.textMuted },
  confidence: { ...typography.caption, color: colors.textMuted, marginTop: 4 },
  careTip: { ...typography.body, color: colors.text, textAlign: 'center', marginTop: 10 },
  body: { ...typography.body, color: colors.textMuted, textAlign: 'center' },
  candidatesLabel: {
    ...typography.caption,
    color: colors.textMuted,
    marginTop: 8,
    alignSelf: 'flex-start',
  },
  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 4 },
  chip: {
    backgroundColor: colors.background,
    borderRadius: 12,
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderWidth: 1,
    borderColor: colors.border,
  },
  chipText: { ...typography.caption, color: colors.text },
  toggleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    paddingHorizontal: 4,
  },
  toggleLabel: { ...typography.body, color: colors.text, flex: 1 },
});
