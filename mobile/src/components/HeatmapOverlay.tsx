import React, { useMemo } from 'react';
import { Canvas, Image as SkiaImageComponent, Skia, AlphaType, ColorType } from '@shopify/react-native-skia';
import { StyleSheet } from 'react-native';

type Props = {
  grid: number[][]; // values already ReLU'd and normalized to [0, 1]
  width: number;
  height: number;
};

// Simple "hot" colormap: transparent -> blue -> green -> yellow -> red, with
// alpha rising alongside intensity so cold/irrelevant regions stay unobtrusive.
function colorForValue(v: number): [number, number, number, number] {
  const stops: [number, number, number, number][] = [
    [0, 90, 220, 0],
    [0, 200, 160, 140],
    [255, 220, 0, 190],
    [230, 60, 20, 220],
  ];
  const scaled = Math.min(0.999, Math.max(0, v)) * (stops.length - 1);
  const i = Math.floor(scaled);
  const t = scaled - i;
  const a = stops[i];
  const b = stops[Math.min(i + 1, stops.length - 1)];
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
    Math.round(a[3] + (b[3] - a[3]) * t),
  ];
}

export function HeatmapOverlay({ grid, width, height }: Props) {
  const skImage = useMemo(() => {
    const size = grid.length;
    const pixels = new Uint8Array(size * size * 4);
    for (let row = 0; row < size; row++) {
      for (let col = 0; col < size; col++) {
        const [r, g, b, a] = colorForValue(grid[row][col]);
        const offset = (row * size + col) * 4;
        pixels[offset] = r;
        pixels[offset + 1] = g;
        pixels[offset + 2] = b;
        pixels[offset + 3] = a;
      }
    }
    const data = Skia.Data.fromBytes(pixels);
    return Skia.Image.MakeImage(
      { width: size, height: size, colorType: ColorType.RGBA_8888, alphaType: AlphaType.Unpremul },
      data,
      size * 4,
    );
  }, [grid]);

  if (!skImage) return null;

  return (
    <Canvas style={[styles.canvas, { width, height }]} pointerEvents="none">
      <SkiaImageComponent image={skImage} x={0} y={0} width={width} height={height} fit="fill" />
    </Canvas>
  );
}

const styles = StyleSheet.create({
  canvas: {
    position: 'absolute',
    top: 0,
    left: 0,
  },
});
