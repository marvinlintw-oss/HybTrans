// src/workers/diarization.worker.ts
import { Speaker } from '../types';

interface SpeechUnit {
  startSec: number;
  endSec: number;
  startSample: number;
  endSample: number;
  f0Hz: number;
  spectralCentroid: number;
  mfccVec: number[];
  speakerId?: string;
}

const SPEAKER_PALETTE = [
  '#2563eb', '#db2777', '#059669', '#d97706', '#7c3aed',
  '#0891b2', '#dc2626', '#4f46e5', '#65a30d', '#ea580c',
];

function computeMagnitudeSpectrum512(frame: Float32Array): Float32Array {
  const N = 512;
  const real = new Float32Array(N);
  const imag = new Float32Array(N);

  for (let i = 0; i < Math.min(N, frame.length); i++) {
    const hann = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
    real[i] = frame[i] * hann;
  }

  let j = 0;
  for (let i = 0; i < N; i++) {
    if (i < j) {
      const tr = real[i];
      real[i] = real[j];
      real[j] = tr;
    }
    let m = N >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  for (let s = 1; s <= 9; s++) {
    const m = 1 << s;
    const halfM = m >> 1;
    const theta = (-2 * Math.PI) / m;
    const wStepR = Math.cos(theta);
    const wStepI = Math.sin(theta);

    for (let k = 0; k < N; k += m) {
      let wR = 1.0;
      let wI = 0.0;
      for (let mIdx = 0; mIdx < halfM; mIdx++) {
        const idx1 = k + mIdx;
        const idx2 = idx1 + halfM;
        const tR = wR * real[idx2] - wI * imag[idx2];
        const tI = wR * imag[idx2] + wI * real[idx2];
        real[idx2] = real[idx1] - tR;
        imag[idx2] = imag[idx1] - tI;
        real[idx1] += tR;
        imag[idx1] += tI;
        const nextWR = wR * wStepR - wI * wStepI;
        wI = wR * wStepI + wI * wStepR;
        wR = nextWR;
      }
    }
  }

  const mag = new Float32Array(N / 2);
  for (let i = 0; i < N / 2; i++) {
    mag[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
  }
  return mag;
}

function estimateFramePitchHz(frame: Float32Array, sampleRate: number): number {
  let energy = 0;
  for (let i = 0; i < frame.length; i += 2) energy += frame[i] * frame[i];
  const rms = Math.sqrt((energy * 2) / frame.length);
  if (rms < 0.009) return 0;

  const minLag = Math.floor(sampleRate / 320);
  const maxLag = Math.floor(sampleRate / 75);
  let bestLag = 0;
  let maxCorr = 0;

  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0;
    let normA = 0;
    let normB = 0;
    const len = frame.length - lag;
    for (let i = 0; i < len; i += 2) {
      const a = frame[i];
      const b = frame[i + lag];
      sum += a * b;
      normA += a * a;
      normB += b * b;
    }
    const denom = Math.sqrt(normA * normB);
    const corr = denom > 1e-6 ? sum / denom : 0;
    if (corr > maxCorr) {
      maxCorr = corr;
      bestLag = lag;
    }
  }

  return maxCorr >= 0.46 && bestLag > 0 ? sampleRate / bestLag : 0;
}

/**
 * 支援「前 90 秒黃金校準期 3 倍高密度採樣 (highDensityBoost)」之深度聲紋提取
 */
function extractDeepSpeakerFeatures(
  samples: Float32Array,
  startSample: number,
  endSample: number,
  sampleRate: number,
  highDensityBoost = false
): { f0Hz: number; spectralCentroid: number; mfccVec: number[] } {
  const frameSize = 512;
  const totalLen = Math.max(1, endSample - startSample);
  // 前 90 秒黃金校準期每 8ms~16ms 密集採樣，徹底拉開同性別不同女聲的共振峰差異！
  const maxFramesToSample = highDensityBoost ? 180 : 75;
  const hopSize = Math.max(128, Math.floor(totalLen / maxFramesToSample));

  const pitches: number[] = [];
  const numMelBands = 20;
  const melAccum = new Float32Array(numMelBands);
  let centroidAccum = 0;
  let validFrames = 0;

  const melBinBounds = [
    3, 5, 7, 10, 13, 17, 21, 26, 32, 39, 47, 56, 67, 79, 93, 109, 128, 150, 176, 206, 240,
  ];

  for (let pos = startSample; pos + frameSize <= endSample; pos += hopSize) {
    const frame = samples.subarray(pos, pos + frameSize);
    let energy = 0;
    for (let i = 0; i < frameSize; i += 2) energy += frame[i] * frame[i];
    if (energy < 1e-4) continue;

    const f0 = estimateFramePitchHz(frame, sampleRate);
    if (f0 > 0) pitches.push(f0);

    const mag = computeMagnitudeSpectrum512(frame);
    validFrames++;

    let weightedFreq = 0;
    let totalMag = 0;
    for (let b = 2; b < 240; b++) {
      weightedFreq += b * mag[b];
      totalMag += mag[b];
    }
    if (totalMag > 1e-5) {
      centroidAccum += (weightedFreq / totalMag) * (sampleRate / 512);
    }

    const frameLogMel = new Float32Array(numMelBands);
    let melMean = 0;
    for (let m = 0; m < numMelBands; m++) {
      const bStart = melBinBounds[m];
      const bEnd = melBinBounds[m + 1];
      let bandEnergy = 0;
      for (let b = bStart; b < bEnd; b++) {
        bandEnergy += mag[b] * mag[b];
      }
      const logE = Math.log10(bandEnergy / Math.max(1, bEnd - bStart) + 1e-7);
      frameLogMel[m] = logE;
      melMean += logE;
    }
    melMean /= numMelBands;

    for (let m = 0; m < numMelBands; m++) {
      melAccum[m] += frameLogMel[m] - melMean;
    }
  }

  pitches.sort((a, b) => a - b);
  const f0Hz =
    pitches.length >= 3
      ? pitches[Math.floor(pitches.length / 2)]
      : pitches.length > 0
      ? pitches[0]
      : 155;

  const spectralCentroid = validFrames > 0 ? centroidAccum / validFrames : 1600;
  const avgLogMel = Array.from(melAccum).map((v) => (validFrames > 0 ? v / validFrames : 0));

  const mfccVec: number[] = [];
  for (let k = 1; k <= 16; k++) {
    let sum = 0;
    for (let n = 0; n < numMelBands; n++) {
      sum += avgLogMel[n] * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * numMelBands));
    }
    mfccVec.push(sum);
  }
  mfccVec.push((avgLogMel[2] + avgLogMel[3]) - (avgLogMel[7] + avgLogMel[8]));
  mfccVec.push((avgLogMel[5] + avgLogMel[6]) - (avgLogMel[11] + avgLogMel[12]));
  mfccVec.push((avgLogMel[14] + avgLogMel[15]) - (avgLogMel[1] + avgLogMel[2]));
  mfccVec.push((spectralCentroid - 1500) / 500);

  return { f0Hz, spectralCentroid, mfccVec };
}

function computeSpeakerDistance(
  f0A: number,
  vecA: number[],
  f0B: number,
  vecB: number[],
  strictGenderGate = true
): number {
  const minF0 = Math.min(f0A, f0B);
  const absDiff = Math.abs(f0A - f0B);
  const relPitchDiff = minF0 > 0 ? absDiff / minF0 : 0;

  if (strictGenderGate && (relPitchDiff > 0.26 || absDiff > 40)) {
    return 999.0;
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;
  const len = Math.min(vecA.length, vecB.length);
  for (let i = 0; i < len; i++) {
    const a = vecA[i] || 0;
    const b = vecB[i] || 0;
    dot += a * b;
    normA += a * a;
    normB += b * b;
  }
  const cosSim = normA > 1e-6 && normB > 1e-6 ? dot / (Math.sqrt(normA) * Math.sqrt(normB)) : 0;
  const mfccDist = Math.max(0, 1 - cosSim);

  return mfccDist * 0.78 + relPitchDiff * 0.82;
}

function extractFineSpeechUnits(
  samples: Float32Array,
  sampleRate: number,
  chunkStartSec: number
): SpeechUnit[] {
  const winSamples = Math.floor(sampleRate * 0.1); // 100ms
  const totalWins = Math.floor(samples.length / winSamples);
  if (totalWins === 0) return [];

  const energies = new Float32Array(totalWins);
  for (let w = 0; w < totalWins; w++) {
    let sum = 0;
    const offset = w * winSamples;
    for (let i = 0; i < winSamples; i += 2) {
      const s = samples[offset + i];
      sum += s * s;
    }
    energies[w] = Math.sqrt((sum * 2) / winSamples);
  }

  const sortedE = Array.from(energies).sort((a, b) => a - b);
  const noiseFloor = sortedE[Math.floor(sortedE.length * 0.18)] || 0.004;
  const vadThreshold = Math.max(0.008, noiseFloor * 2.0);

  const units: SpeechUnit[] = [];
  let inSpeech = false;
  let startW = 0;
  let silenceCount = 0;

  for (let w = 0; w < totalWins; w++) {
    const currentGlobalSec = chunkStartSec + w * 0.1;
    // 前 90 秒啟動 3 倍高密度校準模式 (Opening Calibration Boost)
    const isOpeningBoost = currentGlobalSec <= 90;
    const checkIntervalWins = isOpeningBoost ? 4 : 6; // 前 90 秒每 0.4 秒檢測一次音色換人！
    const minDurForSplit = isOpeningBoost ? 0.8 : 1.2;

    if (energies[w] >= vadThreshold) {
      if (!inSpeech) {
        inSpeech = true;
        startW = w;
      }
      silenceCount = 0;

      const durSec = (w - startW) * 0.1;
      if (
        durSec >= minDurForSplit &&
        (w - startW) % checkIntervalWins === 0 &&
        w + 6 < totalWins
      ) {
        const preS = Math.max(startW * winSamples, (w - 6) * winSamples);
        const midS = w * winSamples;
        const postS = Math.min(samples.length, (w + 6) * winSamples);
        const fPre = extractDeepSpeakerFeatures(samples, preS, midS, sampleRate, isOpeningBoost);
        const fPost = extractDeepSpeakerFeatures(samples, midS, postS, sampleRate, isOpeningBoost);
        const dist = computeSpeakerDistance(fPre.f0Hz, fPre.mfccVec, fPost.f0Hz, fPost.mfccVec, true);

        const splitThreshold = isOpeningBoost ? 0.22 : 0.27;
        if (dist > splitThreshold) {
          const sIdx = startW * winSamples;
          const eIdx = w * winSamples;
          const feat = extractDeepSpeakerFeatures(samples, sIdx, eIdx, sampleRate, isOpeningBoost);
          units.push({
            startSec: chunkStartSec + startW * 0.1,
            endSec: chunkStartSec + w * 0.1,
            startSample: sIdx,
            endSample: eIdx,
            f0Hz: feat.f0Hz,
            spectralCentroid: feat.spectralCentroid,
            mfccVec: feat.mfccVec,
          });
          startW = w;
        }
      }
    } else if (inSpeech) {
      silenceCount++;
      if (silenceCount >= 3 || (w - startW) * 0.1 >= 6.0) {
        const endW = w - silenceCount + 1;
        if ((endW - startW) * 0.1 >= 0.25) {
          const sIdx = startW * winSamples;
          const eIdx = Math.min(samples.length, endW * winSamples);
          const feat = extractDeepSpeakerFeatures(samples, sIdx, eIdx, sampleRate, isOpeningBoost);
          units.push({
            startSec: chunkStartSec + startW * 0.1,
            endSec: chunkStartSec + endW * 0.1,
            startSample: sIdx,
            endSample: eIdx,
            f0Hz: feat.f0Hz,
            spectralCentroid: feat.spectralCentroid,
            mfccVec: feat.mfccVec,
          });
        }
        inSpeech = false;
        silenceCount = 0;
      }
    }
  }

  if (inSpeech && (totalWins - startW) * 0.1 >= 0.25) {
    const sIdx = startW * winSamples;
    const feat = extractDeepSpeakerFeatures(samples, sIdx, samples.length, sampleRate, chunkStartSec <= 90);
    units.push({
      startSec: chunkStartSec + startW * 0.1,
      endSec: chunkStartSec + samples.length / sampleRate,
      startSample: sIdx,
      endSample: samples.length,
      f0Hz: feat.f0Hz,
      spectralCentroid: feat.spectralCentroid,
      mfccVec: feat.mfccVec,
    });
  }

  return units;
}

self.onmessage = (event: MessageEvent) => {
  const data = event.data || {};

  let samples: Float32Array | null = null;
  if (data.samples instanceof Float32Array) {
    samples = data.samples;
  } else if (data.samples instanceof ArrayBuffer) {
    samples = new Float32Array(data.samples);
  }

  const sampleRate = Number(data.sampleRate) || 16000;
  const chunkStartSec = Number(data.chunkStartSec ?? 0);
  const minSpeakers = Math.max(1, Math.min(10, Number(data.minSpeakers) || 2));
  const maxSpeakers = Math.max(minSpeakers, Math.min(10, Number(data.maxSpeakers) || 6));
  const existingSpeakers: Speaker[] = Array.isArray(data.existingSpeakers)
    ? data.existingSpeakers
    : [];

  if (!samples || samples.length === 0) {
    self.postMessage({
      type: 'DIARIZE_SUCCESS',
      turns: [],
      asrWindows: [],
      updatedSpeakers: existingSpeakers,
    });
    return;
  }

  try {
    const units = extractFineSpeechUnits(samples, sampleRate, chunkStartSec);
    if (units.length === 0) {
      self.postMessage({
        type: 'DIARIZE_SUCCESS',
        turns: [],
        asrWindows: [],
        updatedSpeakers: existingSpeakers,
      });
      return;
    }

    const dim = 20;
    for (let d = 0; d < dim; d++) {
      let mean = 0;
      for (const u of units) mean += u.mfccVec[d];
      mean /= units.length;
      let std = 0;
      for (const u of units) std += (u.mfccVec[d] - mean) ** 2;
      std = Math.sqrt(std / units.length) || 1;
      for (const u of units) u.mfccVec[d] = (u.mfccVec[d] - mean) / std;
    }

    const anchorIndices: number[] = [];
    const shortIndices: number[] = [];
    units.forEach((u, idx) => {
      if (u.endSec - u.startSec >= 1.4) {
        anchorIndices.push(idx);
      } else {
        shortIndices.push(idx);
      }
    });

    const activeAnchors =
      anchorIndices.length >= minSpeakers ? anchorIndices : units.map((_, i) => i);

    interface ClusterNode {
      unitIndices: number[];
      f0Hz: number;
      mfccVec: number[];
      totalDurSec: number;
    }

    const clusters: ClusterNode[] = activeAnchors.map((idx) => ({
      unitIndices: [idx],
      f0Hz: units[idx].f0Hz,
      mfccVec: [...units[idx].mfccVec],
      totalDurSec: units[idx].endSec - units[idx].startSec,
    }));

    const updateClusterCentroid = (c: ClusterNode) => {
      let weightedF0 = 0;
      const weightedVec = new Array(dim).fill(0);
      let durSum = 0;
      for (const idx of c.unitIndices) {
        const dur = units[idx].endSec - units[idx].startSec;
        durSum += dur;
        weightedF0 += units[idx].f0Hz * dur;
        for (let d = 0; d < dim; d++) weightedVec[d] += units[idx].mfccVec[d] * dur;
      }
      c.totalDurSec = durSum;
      c.f0Hz = durSum > 0 ? weightedF0 / durSum : c.f0Hz;
      c.mfccVec = weightedVec.map((v) => (durSum > 0 ? v / durSum : 0));
    };

    const maxIterations = clusters.length + 5;
    for (let iter = 0; iter < maxIterations && clusters.length > minSpeakers; iter++) {
      let bestI = -1;
      let bestJ = -1;
      let minDist = Infinity;

      for (let i = 0; i < clusters.length; i++) {
        for (let j = i + 1; j < clusters.length; j++) {
          const dist = computeSpeakerDistance(
            clusters[i].f0Hz,
            clusters[i].mfccVec,
            clusters[j].f0Hz,
            clusters[j].mfccVec,
            clusters.length <= maxSpeakers + 2
          );
          if (dist < minDist) {
            minDist = dist;
            bestI = i;
            bestJ = j;
          }
        }
      }

      if (bestI === -1 || bestJ === -1) break;
      if (clusters.length <= maxSpeakers && minDist > 0.41) break;
      if (minDist >= 900 && clusters.length <= maxSpeakers) break;

      clusters[bestI].unitIndices.push(...clusters[bestJ].unitIndices);
      updateClusterCentroid(clusters[bestI]);
      clusters.splice(bestJ, 1);
    }

    const totalAnchorDur = clusters.reduce((acc, c) => acc + c.totalDurSec, 0);
    for (let step = 0; step < 25 && clusters.length > minSpeakers; step++) {
      const tinyIdx = clusters.findIndex((c) => c.totalDurSec / (totalAnchorDur || 1) < 0.045);
      if (tinyIdx === -1) break;

      let mergeTarget = -1;
      let bestDist = Infinity;
      for (let i = 0; i < clusters.length; i++) {
        if (i === tinyIdx) continue;
        const d = computeSpeakerDistance(
          clusters[tinyIdx].f0Hz,
          clusters[tinyIdx].mfccVec,
          clusters[i].f0Hz,
          clusters[i].mfccVec,
          false
        );
        if (d < bestDist) {
          bestDist = d;
          mergeTarget = i;
        }
      }
      if (mergeTarget === -1) break;
      clusters[mergeTarget].unitIndices.push(...clusters[tinyIdx].unitIndices);
      updateClusterCentroid(clusters[mergeTarget]);
      clusters.splice(tinyIdx, 1);
    }

    clusters.sort((a, b) => b.totalDurSec - a.totalDurSec);
    if (clusters.length > maxSpeakers) {
      const overflow = clusters.splice(maxSpeakers);
      for (const ov of overflow) {
        clusters[0].unitIndices.push(...ov.unitIndices);
      }
      updateClusterCentroid(clusters[0]);
    }

    const updatedSpeakers: Speaker[] = existingSpeakers.map((sp) => ({
      id: sp.id,
      name: sp.name,
      color: sp.color,
      centroid: sp.centroid ? [...sp.centroid] : undefined,
      mergedFrom: Array.isArray(sp.mergedFrom) ? [...sp.mergedFrom] : [],
    }));

    const usedSpeakerIds = new Set<string>();
    const clusterSpeakerIds: string[] = new Array(clusters.length).fill('');

    for (let cIdx = 0; cIdx < clusters.length; cIdx++) {
      const c = clusters[cIdx];
      let matchedSp: Speaker | null = null;
      let bestMatchDist = Infinity;

      for (const sp of updatedSpeakers) {
        if (usedSpeakerIds.has(sp.id) || !sp.centroid || sp.centroid.length < 21) continue;
        const spF0 = sp.centroid[0];
        const spVec = sp.centroid.slice(1, 21);
        const dist = computeSpeakerDistance(c.f0Hz, c.mfccVec, spF0, spVec, true);
        if (dist < bestMatchDist) {
          bestMatchDist = dist;
          matchedSp = sp;
        }
      }

      if (matchedSp && bestMatchDist < 0.52) {
        usedSpeakerIds.add(matchedSp.id);
        clusterSpeakerIds[cIdx] = matchedSp.id;
        const oldF0 = matchedSp.centroid![0];
        const oldVec = matchedSp.centroid!.slice(1, 21);
        matchedSp.centroid = [
          oldF0 * 0.75 + c.f0Hz * 0.25,
          ...oldVec.map((v, idx) => v * 0.75 + c.mfccVec[idx] * 0.25),
        ];
      } else if (updatedSpeakers.length < maxSpeakers) {
        const nextNum = updatedSpeakers.length + 1;
        const newId = `Speaker_${String(nextNum).padStart(2, '0')}`;
        const genderHint = c.f0Hz >= 168 ? '女聲' : c.f0Hz <= 152 ? '男聲' : '語者';
        const newSp: Speaker = {
          id: newId,
          name: `語者 ${String(nextNum).padStart(2, '0')} (${genderHint})`,
          color: SPEAKER_PALETTE[(nextNum - 1) % SPEAKER_PALETTE.length],
          centroid: [c.f0Hz, ...c.mfccVec],
          mergedFrom: [],
        };
        updatedSpeakers.push(newSp);
        usedSpeakerIds.add(newId);
        clusterSpeakerIds[cIdx] = newId;
      } else {
        let fallbackSp = updatedSpeakers[0];
        let minD = Infinity;
        for (const sp of updatedSpeakers) {
          const spF0 = sp.centroid?.[0] || 155;
          const spVec = sp.centroid?.slice(1, 21) || c.mfccVec;
          const d = computeSpeakerDistance(c.f0Hz, c.mfccVec, spF0, spVec, false);
          if (d < minD) {
            minD = d;
            fallbackSp = sp;
          }
        }
        clusterSpeakerIds[cIdx] = fallbackSp.id;
      }

      for (const uIdx of c.unitIndices) {
        units[uIdx].speakerId = clusterSpeakerIds[cIdx];
      }
    }

    for (const sIdx of shortIndices) {
      if (units[sIdx].speakerId) continue;
      const u = units[sIdx];
      let bestSpId = clusterSpeakerIds[0] || 'Speaker_01';
      let bestScore = Infinity;

      for (let cIdx = 0; cIdx < clusters.length; cIdx++) {
        const c = clusters[cIdx];
        const dist = computeSpeakerDistance(u.f0Hz, u.mfccVec, c.f0Hz, c.mfccVec, false);
        const prevU = sIdx > 0 ? units[sIdx - 1] : null;
        const continuityBonus =
          prevU && prevU.speakerId === clusterSpeakerIds[cIdx] && u.startSec - prevU.endSec < 1.5
            ? -0.18
            : 0;
        const score = dist + continuityBonus;
        if (score < bestScore) {
          bestScore = score;
          bestSpId = clusterSpeakerIds[cIdx];
        }
      }
      u.speakerId = bestSpId;
    }

    // 維特比乒乓平滑濾波器 (保留真實搶話，撫平同一人換氣波動)
    for (let i = 1; i < units.length - 1; i++) {
      const prev = units[i - 1];
      const curr = units[i];
      const next = units[i + 1];
      const currDur = curr.endSec - curr.startSec;

      if (
        prev.speakerId &&
        prev.speakerId === next.speakerId &&
        curr.speakerId !== prev.speakerId &&
        currDur <= 1.8 &&
        next.startSec - prev.endSec <= 3.0
      ) {
        const f0DiffRatio = Math.abs(curr.f0Hz - prev.f0Hz) / Math.max(80, prev.f0Hz);
        if (f0DiffRatio < 0.22) {
          curr.speakerId = prev.speakerId;
        }
      }
    }

    // 保留極細粒度聲紋時間軸 (不把短於 2 秒的轉折吃掉)，供 ASR 子句級對齊器精準查表！
    const fineTimelineTurns: Array<{
      start: number;
      end: number;
      speakerId: string;
      f0Hz: number;
    }> = units.map((u) => ({
      start: Number(u.startSec.toFixed(2)),
      end: Number(u.endSec.toFixed(2)),
      speakerId: u.speakerId || 'Speaker_01',
      f0Hz: Math.round(u.f0Hz),
    }));

    // 前 90 秒採用更精細的 14~18 秒 ASR 視窗，減少長視窗內部時間戳漂移；90 秒後採用 22~25 秒視窗
    const asrWindows: Array<{ start: number; end: number }> = [];
    let winStart = fineTimelineTurns[0].start;
    let lastEnd = fineTimelineTurns[0].end;

    for (let i = 1; i < fineTimelineTurns.length; i++) {
      const cur = fineTimelineTurns[i];
      const gap = cur.start - lastEnd;
      const currentWinDur = lastEnd - winStart;
      const targetWinLen = winStart < 90 ? 14.0 : 20.0;
      const maxWinLen = winStart < 90 ? 18.0 : 25.0;

      if (
        (currentWinDur >= targetWinLen && gap >= 0.3) ||
        currentWinDur >= maxWinLen ||
        gap >= 2.2
      ) {
        asrWindows.push({
          start: Math.max(chunkStartSec, Number((winStart - 0.1).toFixed(2))),
          end: Number((lastEnd + 0.15).toFixed(2)),
        });
        winStart = cur.start;
      }
      lastEnd = cur.end;
    }

    asrWindows.push({
      start: Math.max(chunkStartSec, Number((winStart - 0.1).toFixed(2))),
      end: Number((lastEnd + 0.15).toFixed(2)),
    });

    self.postMessage({
      type: 'DIARIZE_SUCCESS',
      turns: fineTimelineTurns,
      asrWindows,
      updatedSpeakers,
    });
  } catch (err) {
    self.postMessage({
      type: 'DIARIZE_ERROR',
      error: (err as Error).message || '語者分離發生未知錯誤',
    });
  }
};