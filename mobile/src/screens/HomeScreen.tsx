import React, { useCallback, useState } from 'react';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { colors } from '../theme/colors';
import { typography } from '../theme/typography';
import { HomeStackParamList, RootTabParamList } from '../navigation/types';
import { HistoryEntry, loadHistory } from '../store/historyStore';

type Props = NativeStackScreenProps<HomeStackParamList, 'Home'>;

function greeting(): string {
  const hour = new Date().getHours();
  if (hour < 6) return 'Доброї ночі';
  if (hour < 12) return 'Доброго ранку';
  if (hour < 18) return 'Доброго дня';
  return 'Доброго вечора';
}

export function HomeScreen({ navigation }: Props) {
  const [lastScan, setLastScan] = useState<HistoryEntry | null>(null);
  const [scanCount, setScanCount] = useState(0);

  useFocusEffect(
    useCallback(() => {
      const history = loadHistory();
      setLastScan(history[0] ?? null);
      setScanCount(history.length);
    }, []),
  );

  const goToHistory = () => {
    navigation.getParent<BottomTabNavigationProp<RootTabParamList>>()?.navigate('HistoryTab');
  };

  return (
    <View style={styles.container}>
      <Text style={styles.decorTopLeft}>🍃</Text>
      <Text style={styles.decorTopRight}>🌿</Text>

      <View style={styles.hero}>
        <Text style={styles.greeting}>{greeting()}!</Text>
        <View style={styles.badge}>
          <Text style={styles.emoji}>🌿</Text>
        </View>
        <Text style={styles.title}>CropDoctor</Text>
        <Text style={styles.subtitle}>
          Сфотографуйте рослину і визначіть захворювання прямо в телефоні
        </Text>
        {scanCount > 0 && (
          <Text style={styles.scanCount}>
            Всього перевірено рослин: {scanCount}
          </Text>
        )}
      </View>

      {lastScan && (
        <Pressable onPress={goToHistory}>
          <Card style={styles.lastScanCard}>
            <Image source={{ uri: lastScan.imagePath }} style={styles.lastScanThumb} />
            <View style={styles.lastScanText}>
              <Text style={styles.lastScanLabel}>Останній скан</Text>
              <Text style={styles.lastScanTitle} numberOfLines={1}>
                {lastScan.speciesUk} · {lastScan.healthy ? 'здорова' : lastScan.conditionUk}
              </Text>
            </View>
            <Text style={styles.chevron}>›</Text>
          </Card>
        </Pressable>
      )}

      <View style={styles.actions}>
        <Button
          label="📷  Сфотографувати рослину"
          onPress={() => navigation.navigate('Capture', { source: 'camera' })}
        />
        <Button
          label="🖼️  Обрати з галереї"
          variant="secondary"
          onPress={() => navigation.navigate('Capture', { source: 'gallery' })}
          style={styles.secondaryButton}
        />
        {/* <Pressable onPress={() => navigation.navigate('DemoSamples')} style={styles.demoLink}>
          <Text style={styles.demoLinkText}>🧪 Спробувати на тестових прикладах</Text>
        </Pressable> */}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    justifyContent: 'space-between',
    padding: 24,
    paddingTop: 64,
  },
  decorTopLeft: {
    position: 'absolute',
    top: 24,
    left: 8,
    fontSize: 28,
    opacity: 0.18,
    transform: [{ rotate: '-25deg' }],
  },
  decorTopRight: {
    position: 'absolute',
    top: 90,
    right: 12,
    fontSize: 22,
    opacity: 0.18,
    transform: [{ rotate: '20deg' }],
  },
  hero: {
    alignItems: 'center',
    gap: 8,
  },
  greeting: {
    ...typography.subheading,
    color: colors.secondary,
    fontSize: 16,
  },
  badge: {
    width: 108,
    height: 108,
    borderRadius: 54,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: 4,
    shadowColor: colors.primary,
    shadowOpacity: 0.12,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 6 },
    elevation: 3,
  },
  emoji: {
    fontSize: 56,
  },
  title: {
    ...typography.heading,
    fontSize: 32,
    color: colors.primary,
  },
  subtitle: {
    ...typography.body,
    color: colors.textMuted,
    textAlign: 'center',
    paddingHorizontal: 12,
  },
  scanCount: {
    ...typography.caption,
    color: colors.secondary,
    marginTop: 4,
  },
  lastScanCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingVertical: 12,
  },
  lastScanThumb: {
    width: 48,
    height: 48,
    borderRadius: 12,
    backgroundColor: colors.border,
  },
  lastScanText: { flex: 1 },
  lastScanLabel: { ...typography.caption, color: colors.textMuted },
  lastScanTitle: { ...typography.subheading, fontSize: 15, color: colors.text, marginTop: 2 },
  chevron: { ...typography.heading, fontSize: 22, color: colors.textMuted },
  actions: {
    gap: 12,
    paddingBottom: 24,
  },
  secondaryButton: {
    marginTop: 4,
  },
  demoLink: {
    alignItems: 'center',
    paddingVertical: 10,
  },
  demoLinkText: {
    ...typography.body,
    color: colors.secondary,
    fontSize: 14,
  },
});
