// Nunito (headings) + Inter (body) are bundled as static-weight TTFs
// (instantiated from the variable Google Fonts sources via fonttools, see
// mobile/assets/fonts) and shipped as raw Android assets under
// android/app/src/main/assets/fonts/ — the filename must match fontFamily
// exactly for React Native's Android font resolution to find it.
export const typography = {
  heading: {
    fontFamily: 'Nunito-Bold',
    fontSize: 24,
    lineHeight: 30,
  },
  subheading: {
    fontFamily: 'Nunito-SemiBold',
    fontSize: 18,
    lineHeight: 24,
  },
  body: {
    fontFamily: 'Inter-Regular',
    fontSize: 15,
    lineHeight: 21,
  },
  caption: {
    fontFamily: 'Inter-Regular',
    fontSize: 12,
    lineHeight: 16,
  },
} as const;
