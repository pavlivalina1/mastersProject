import React, { useCallback, useState } from 'react';
import { useFocusEffect } from '@react-navigation/native';
import { FlatList, Image, StyleSheet, Text, View } from 'react-native';
import { colors } from '../theme/colors';
import { typography } from '../theme/typography';
import { HistoryEntry, loadHistory } from '../store/historyStore';

export function HistoryScreen() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);

  useFocusEffect(
    useCallback(() => {
      setEntries(loadHistory());
    }, []),
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Історія сканів</Text>
      {entries.length === 0 ? (
        <Text style={styles.empty}>Тут з'являться ваші попередні результати</Text>
      ) : (
        <FlatList
          data={entries}
          keyExtractor={item => item.id}
          contentContainerStyle={styles.list}
          renderItem={({ item }) => (
            <View style={styles.row}>
              <Image source={{ uri: item.imagePath }} style={styles.thumb} />
              <View style={styles.rowText}>
                <Text style={styles.species}>{item.speciesUk}</Text>
                <Text style={styles.condition}>
                  {item.healthy ? 'Здорова' : item.conditionUk} ·{' '}
                  {Math.round(item.confidence * 100)}%
                </Text>
                <Text style={styles.date}>
                  {new Date(item.timestamp).toLocaleString('uk-UA')}
                </Text>
              </View>
            </View>
          )}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background, padding: 24, paddingTop: 64, gap: 12 },
  title: { ...typography.heading, color: colors.primary },
  empty: { ...typography.body, color: colors.textMuted },
  list: { gap: 12 },
  row: {
    flexDirection: 'row',
    gap: 12,
    backgroundColor: colors.surface,
    borderRadius: 16,
    padding: 10,
  },
  thumb: { width: 56, height: 56, borderRadius: 12, backgroundColor: colors.border },
  rowText: { flex: 1, justifyContent: 'center' },
  species: { ...typography.subheading, fontSize: 15, color: colors.text },
  condition: { ...typography.body, fontSize: 13, color: colors.textMuted },
  date: { ...typography.caption, color: colors.textMuted, marginTop: 2 },
});
