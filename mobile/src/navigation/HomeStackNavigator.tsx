import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { colors } from '../theme/colors';
import { HomeScreen } from '../screens/HomeScreen';
import { CaptureScreen } from '../screens/CaptureScreen';
import { DemoSamplesScreen } from '../screens/DemoSamplesScreen';
import { ResultScreen } from '../screens/ResultScreen';
import { HomeStackParamList } from './types';

const Stack = createNativeStackNavigator<HomeStackParamList>();

export function HomeStackNavigator() {
  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: { backgroundColor: colors.background },
        headerTintColor: colors.primary,
        headerShadowVisible: false,
      }}>
      <Stack.Screen name="Home" component={HomeScreen} options={{ headerShown: false }} />
      <Stack.Screen name="Capture" component={CaptureScreen} options={{ title: 'Фото листка' }} />
      <Stack.Screen
        name="DemoSamples"
        component={DemoSamplesScreen}
        options={{ title: 'Тестові приклади' }}
      />
      <Stack.Screen name="Result" component={ResultScreen} options={{ title: 'Результат' }} />
    </Stack.Navigator>
  );
}
