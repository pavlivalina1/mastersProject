import { useCallback } from 'react';
import { imageUriToTensor } from './preprocess';
import { runInference } from './session';
import { labelById } from './labels';
import { CAM_GRID_SIZE, CONFIDENCE_THRESHOLD } from './constants';

export type ClassificationCandidate = {
  labelId: number;
  speciesUk: string;
  conditionUk: string;
  healthy: boolean;
  confidence: number;
  careTipUk: string;
};

export type ClassificationResult = {
  top: ClassificationCandidate;
  top3: ClassificationCandidate[];
  isConfident: boolean;
  /** CAM heatmap for the top class, ReLU'd and min-max normalized to [0,1]. */
  camGrid: number[][];
};

function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (const v of logits) {
    if (v > max) max = v;
  }
  let sum = 0;
  const exps = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] - max);
    sum += exps[i];
  }
  for (let i = 0; i < exps.length; i++) {
    exps[i] /= sum;
  }
  return exps;
}

function toCandidate(labelId: number, confidence: number): ClassificationCandidate {
  const label = labelById(labelId);
  return {
    labelId,
    speciesUk: label.species_uk,
    conditionUk: label.condition_uk,
    healthy: label.healthy,
    confidence,
    careTipUk: label.care_tip_uk,
  };
}

// cam is [num_classes * gridSize * gridSize], row-major per class (see
// src/export_mobile.py's _WithCAM / einsum('bkhw,ck->bchw', ...)).
function extractCamGrid(cam: Float32Array, classIdx: number, gridSize: number): number[][] {
  const cellCount = gridSize * gridSize;
  const offset = classIdx * cellCount;

  let max = -Infinity;
  const relu = new Float32Array(cellCount);
  for (let i = 0; i < cellCount; i++) {
    const v = Math.max(0, cam[offset + i]); // ReLU: keep only positive support for this class
    relu[i] = v;
    if (v > max) max = v;
  }

  const grid: number[][] = [];
  for (let row = 0; row < gridSize; row++) {
    const rowValues: number[] = [];
    for (let col = 0; col < gridSize; col++) {
      const v = relu[row * gridSize + col];
      rowValues.push(max > 0 ? v / max : 0);
    }
    grid.push(rowValues);
  }
  return grid;
}

export function useCropClassifier() {
  const classify = useCallback(async (imageUri: string): Promise<ClassificationResult> => {
    const tensor = await imageUriToTensor(imageUri);
    const { logits, cam } = await runInference(tensor);
    const probs = softmax(logits);

    const indices = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
    const top3 = indices.slice(0, 3).map(idx => toCandidate(idx, probs[idx]));
    const camGrid = extractCamGrid(cam, top3[0].labelId, CAM_GRID_SIZE);

    return {
      top: top3[0],
      top3,
      isConfident: top3[0].confidence >= CONFIDENCE_THRESHOLD,
      camGrid,
    };
  }, []);

  return { classify };
}
