import { createMMKV } from 'react-native-mmkv';

export type HistoryEntry = {
  id: string;
  timestamp: number;
  imagePath: string;
  speciesUk: string;
  conditionUk: string;
  healthy: boolean;
  confidence: number;
};

const storage = createMMKV({ id: 'crop-doctor-history' });
const ENTRIES_KEY = 'entries';

export function loadHistory(): HistoryEntry[] {
  const raw = storage.getString(ENTRIES_KEY);
  return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
}

export function addHistoryEntry(entry: HistoryEntry): HistoryEntry[] {
  const entries = [entry, ...loadHistory()];
  storage.set(ENTRIES_KEY, JSON.stringify(entries));
  return entries;
}
