export const colors = {
  background: '#FAF8F2',
  surface: '#FFFFFF',
  primary: '#2E5339',
  primaryDark: '#1E3B27',
  secondary: '#7FB069',
  accent: '#E4A62F',
  danger: '#C1502E',
  text: '#26332B',
  textMuted: '#6B7A70',
  border: '#E4E1D6',
} as const;

export type ColorToken = keyof typeof colors;
