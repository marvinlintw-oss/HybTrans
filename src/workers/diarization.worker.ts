// src/workers/diarization.worker.ts
import {
  DiarizationConfig,
  FineDiarizationTurn,
  Speaker,
} from '../types';

const SPEAKER_PALETTE = [
  '#2563eb', '#db2777', '#059669', '#d97706', '#7c3aed',
  '#0891b2', '#dc2626', '#4f46e5', '#65a30d', '#ea580c',
];

const LOCAL_MODEL_CACHE_NAME = 'hybrid-transcriber-local-models-v1';

function resolvePublicRootUrl(relPath: string): string {
  const cleanRel = relPath.replace(/^(\.\/|\/)+/, '');
  const href = self.location.href;
  const baseRoot = href.includes('/src/workers/')
    ? href.split('/src/workers/')[0] + '/'
    : href.includes('/assets/')
    ? href.split('/assets/')[0] + '/'
    : self.location.origin + '/';
  return `${baseRoot}${cleanRel}`;
}

function l2Normalize(vec: Float32Array | number[]): Float32Array {
  let normSq = 0.0;
  for (let i = 0; i < vec.length; i++) {
    normSq += vec[i] * vec[i];
  }
  const invNorm = 1.0 / Math.sqrt(normSq + 1e-12);
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    out[i] = vec[i] * invNorm;
  }
  return out;
}

function cosineSimilarity(a: Float32Array | number[], b: Float32Array | number[]): number {
  const len = Math.min(a.length, b.length);
  if (len === 0) return 0.0;
  let dot = 0.0;
  let nA = 0.0;
  let nB = 0.0;
  for (let i = 0; i < len; i++) {
    const va = a[i] || 0;
    const vb = b[i] || 0;
    dot += va * vb;
    nA += va * va;
    nB += vb * vb;
  }
  const denom = Math.sqrt(nA * nB);
  return denom > 1e-10 ? Math.max(-1.0, Math.min(1.0, dot / denom)) : 0.0;
}

function estimateMedianPitchHz(
  samples: Float32Array,
  startSample: number,
  endSample: number,
  sampleRate: number
): number {
  const frameLen = 512;
  const total = Math.max(1, endSample - startSample);
  const hop = Math.max(256, Math.floor(total / 30));
  const pitches: number[] = [];
  const minLag = Math.floor(sampleRate / 320);
  const maxLag = Math.floor(sampleRate / 75);

  for (let pos = startSample; pos + frameLen <= endSample; pos += hop) {
    let energy = 0.0;
    for (let i = 0; i < frameLen; i += 2) {
      const v = samples[pos + i];
      energy += v * v;
    }
    if (Math.sqrt((energy * 2) / frameLen) < 0.01) continue;

    let bestLag = 0;
    let maxCorr = 0.0;
    for (let lag = minLag; lag <= maxLag; lag += 2) {
      let sum = 0.0;
      let nA = 0.0;
      let nB = 0.0;
      const limit = frameLen - lag;
      for (let i = 0; i < limit; i += 2) {
        const a = samples[pos + i];
        const b = samples[pos + i + lag];
        sum += a * b;
        nA += a * a;
        nB += b * b;
      }
      const corr = nA > 1e-8 && nB > 1e-8 ? sum / Math.sqrt(nA * nB) : 0.0;
      if (corr > maxCorr) {
        maxCorr = corr;
        bestLag = lag;
      }
    }
    if (maxCorr >= 0.45 && bestLag > 0) {
      pitches.push(sampleRate / bestLag);
    }
  }

  if (pitches.length === 0) return 155;
  pitches.sort((a, b) => a - b);
  return pitches[Math.floor(pitches.length / 2)];
}

// ============================================================================
// 1. Bit-Exact Kaldi 80-Dim Fbank (x32768 Int16 Scale + Per-Bin CMN)
// ============================================================================

export class KaldiFbankExtractor {
  private readonly sampleRate = 16000;
  private readonly frameLength = 400; // 25ms @ 16kHz
  private readonly frameShift = 160;  // 10ms @ 16kHz
  private readonly fftSize = 512;
  private readonly numMelBins = 80;
  private readonly preemph = 0.97;
  private readonly hammingWindow: Float32Array;
  private readonly melFilters: Float32Array[];

  constructor() {
    this.hammingWindow = new Float32Array(this.frameLength);
    for (let n = 0; n < this.frameLength; n++) {
      this.hammingWindow[n] =
        0.54 - 0.46 * Math.cos((2.0 * Math.PI * n) / (this.frameLength - 1));
    }
    this.melFilters = this.initKaldiMelFilterbank(20.0, 8000.0);
  }

  private hzToMel(hz: number): number {
    return 1127.0 * Math.log(1.0 + hz / 700.0);
  }

  private initKaldiMelFilterbank(lowFreq: number, highFreq: number): Float32Array[] {
    const numFftBins = this.fftSize / 2 + 1;
    const melLow = this.hzToMel(lowFreq);
    const melHigh = this.hzToMel(highFreq);
    const melDelta = (melHigh - melLow) / (this.numMelBins + 1);
    const fftBinWidth = this.sampleRate / this.fftSize;

    const filters: Float32Array[] = [];
    for (let b = 0; b < this.numMelBins; b++) {
      const leftMel = melLow + b * melDelta;
      const centerMel = melLow + (b + 1) * melDelta;
      const rightMel = melLow + (b + 2) * melDelta;
      const filter = new Float32Array(numFftBins);

      for (let k = 0; k < numFftBins; k++) {
        const freq = k * fftBinWidth;
        const mel = this.hzToMel(freq);
        if (mel > leftMel && mel < rightMel) {
          filter[k] =
            mel <= centerMel
              ? (mel - leftMel) / (centerMel - leftMel)
              : (rightMel - mel) / (rightMel - centerMel);
        }
      }
      filters.push(filter);
    }
    return filters;
  }

  public computeGlobalFbank(pcmFloat32: Float32Array): {
    frames: Float32Array[];
    frameRms: Float32Array;
    numFrames: number;
  } {
    if (pcmFloat32.length < this.frameLength) {
      return { frames: [], frameRms: new Float32Array(0), numFrames: 0 };
    }
    const numFrames =
      1 + Math.floor((pcmFloat32.length - this.frameLength) / this.frameShift);
    const frames: Float32Array[] = new Array(numFrames);
    const frameRms = new Float32Array(numFrames);

    const frameBuf = new Float32Array(this.frameLength);
    const real = new Float32Array(this.fftSize);
    const imag = new Float32Array(this.fftSize);
    const powerSpec = new Float32Array(this.fftSize / 2 + 1);
    const fltEpsilon = 1.1920929e-7;

    for (let t = 0; t < numFrames; t++) {
      const offset = t * this.frameShift;

      let mean = 0.0;
      let rawEnergySq = 0.0;
      for (let n = 0; n < this.frameLength; n++) {
        const rawSample = pcmFloat32[offset + n];
        rawEnergySq += rawSample * rawSample;
        const scaledVal = rawSample * 32768.0; // WeSpeaker Int16 scale
        frameBuf[n] = scaledVal;
        mean += scaledVal;
      }
      mean /= this.frameLength;
      frameRms[t] = Math.sqrt(rawEnergySq / this.frameLength);

      real.fill(0);
      imag.fill(0);
      let prev = frameBuf[0] - mean;
      real[0] = prev * (1.0 - this.preemph) * this.hammingWindow[0];
      for (let n = 1; n < this.frameLength; n++) {
        const curr = frameBuf[n] - mean;
        real[n] = (curr - this.preemph * prev) * this.hammingWindow[n];
        prev = curr;
      }

      this.inPlaceFFT512(real, imag);
      for (let k = 0; k <= this.fftSize / 2; k++) {
        powerSpec[k] = real[k] * real[k] + imag[k] * imag[k];
      }

      const melFrame = new Float32Array(this.numMelBins);
      for (let b = 0; b < this.numMelBins; b++) {
        let energy = 0.0;
        const filter = this.melFilters[b];
        for (let k = 0; k <= this.fftSize / 2; k++) {
          energy += filter[k] * powerSpec[k];
        }
        melFrame[b] = Math.log(Math.max(energy, fltEpsilon));
      }
      frames[t] = melFrame;
    }

    return { frames, frameRms, numFrames };
  }

  public sliceAndApplyCMN(
    allFrames: Float32Array[],
    startFrame: number,
    endFrame: number
  ): { cmnTensor: Float32Array; binMeans: Float32Array } {
    const T = Math.max(1, endFrame - startFrame);
    const cmnTensor = new Float32Array(T * this.numMelBins);
    const binMeans = new Float32Array(this.numMelBins);

    for (let t = startFrame; t < endFrame; t++) {
      const frame = allFrames[t];
      for (let b = 0; b < this.numMelBins; b++) {
        binMeans[b] += frame[b];
      }
    }
    for (let b = 0; b < this.numMelBins; b++) {
      binMeans[b] /= T;
    }

    for (let t = 0; t < T; t++) {
      const frame = allFrames[startFrame + t];
      const rowOffset = t * this.numMelBins;
      for (let b = 0; b < this.numMelBins; b++) {
        cmnTensor[rowOffset + b] = frame[b] - binMeans[b];
      }
    }
    return { cmnTensor, binMeans };
  }

  public computeFormantPreservingTstpEmbedding(
    cmnTensor: Float32Array,
    binMeans: Float32Array,
    numFrames: number
  ): Float32Array {
    const dim = 64;
    const out = new Float32Array(dim);
    if (numFrames <= 0) return out;

    let globalGain = 0.0;
    for (let b = 0; b < 80; b++) globalGain += binMeans[b];
    globalGain /= 80.0;

    const normSpectralEnvelope = new Float32Array(80);
    for (let b = 0; b < 80; b++) {
      normSpectralEnvelope[b] = binMeans[b] - globalGain;
    }

    for (let k = 1; k <= 20; k++) {
      let sum = 0.0;
      for (let b = 0; b < 80; b++) {
        sum +=
          normSpectralEnvelope[b] *
          Math.cos((Math.PI * k * (2 * b + 1)) / 160.0);
      }
      out[k - 1] = sum * 0.12;
    }

    for (let i = 0; i < 24; i++) {
      const b0 = 20 + i * 2;
      out[20 + i] = (normSpectralEnvelope[b0] + normSpectralEnvelope[b0 + 1]) * 0.45;
    }

    const bandStd = new Float32Array(80);
    for (let t = 0; t < numFrames; t++) {
      const off = t * 80;
      for (let b = 0; b < 80; b++) {
        const v = cmnTensor[off + b];
        bandStd[b] += v * v;
      }
    }
    const denom = Math.max(1, numFrames - 1);
    for (let b = 0; b < 80; b++) {
      bandStd[b] = Math.sqrt(bandStd[b] / denom + 1e-7);
    }

    let stdMean = 0.0;
    for (let b = 0; b < 80; b++) stdMean += bandStd[b];
    stdMean /= 80.0;

    for (let i = 0; i < 20; i++) {
      const bStart = i * 4;
      const avgStd =
        (bandStd[bStart] +
          bandStd[bStart + 1] +
          bandStd[bStart + 2] +
          bandStd[bStart + 3]) *
        0.25;
      out[44 + i] = (avgStd - stdMean) * 0.85;
    }

    return l2Normalize(out);
  }

  private inPlaceFFT512(real: Float32Array, imag: Float32Array): void {
    const N = 512;
    let j = 0;
    for (let i = 0; i < N; i++) {
      if (i < j) {
        const tr = real[i];
        real[i] = real[j];
        real[j] = tr;
        const ti = imag[i];
        imag[i] = imag[j];
        imag[j] = ti;
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
      const theta = (-2.0 * Math.PI) / m;
      const wpr = Math.cos(theta);
      const wpi = Math.sin(theta);
      for (let k = 0; k < N; k += m) {
        let wr = 1.0;
        let wi = 0.0;
        for (let l = 0; l < halfM; l++) {
          const idx1 = k + l;
          const idx2 = idx1 + halfM;
          const tr = wr * real[idx2] - wi * imag[idx2];
          const ti = wr * imag[idx2] + wi * real[idx2];
          real[idx2] = real[idx1] - tr;
          imag[idx2] = imag[idx1] - ti;
          real[idx1] += tr;
          imag[idx1] += ti;
          const nextWr = wr * wpr - wi * wpi;
          wi = wr * wpi + wi * wpr;
          wr = nextWr;
        }
      }
    }
  }
}

// ============================================================================
// 2. 純 TypeScript 未知人數自動調參譜聚類 (Auto-Tuning NME-SC)
// ============================================================================

export class AutoTuningSpectralClusterer {
  public static symmetricEigenDecomposition(
    matrix: Float64Array,
    N: number,
    computeVectors: boolean
  ): { eigenvalues: Float64Array; eigenvectors?: Float64Array } {
    const d = new Float64Array(N);
    const e = new Float64Array(N);
    const z = new Float64Array(matrix);

    for (let i = N - 1; i > 0; i--) {
      const l = i - 1;
      let h = 0.0;
      let scale = 0.0;
      if (l > 0) {
        for (let k = 0; k <= l; k++) scale += Math.abs(z[i * N + k]);
        if (scale === 0.0) {
          e[i] = z[i * N + l];
        } else {
          for (let k = 0; k <= l; k++) {
            z[i * N + k] /= scale;
            h += z[i * N + k] * z[i * N + k];
          }
          let f = z[i * N + l];
          const g = f >= 0.0 ? -Math.sqrt(h) : Math.sqrt(h);
          e[i] = scale * g;
          h -= f * g;
          z[i * N + l] = f - g;
          f = 0.0;
          for (let j = 0; j <= l; j++) {
            if (computeVectors) z[j * N + i] = z[i * N + j] / h;
            let g2 = 0.0;
            for (let k = 0; k <= j; k++) g2 += z[j * N + k] * z[i * N + k];
            for (let k = j + 1; k <= l; k++) g2 += z[k * N + j] * z[i * N + k];
            e[j] = g2 / h;
            f += e[j] * z[i * N + j];
          }
          const hh = f / (h + h);
          for (let j = 0; j <= l; j++) {
            f = z[i * N + j];
            e[j] = e[j] - hh * f;
            const g2 = e[j];
            for (let k = 0; k <= j; k++) {
              z[j * N + k] -= f * e[k] + g2 * z[i * N + k];
            }
          }
        }
      } else {
        e[i] = z[i * N + l];
      }
      d[i] = h;
    }
    d[0] = 0.0;
    e[0] = 0.0;
    for (let i = 0; i < N; i++) {
      if (computeVectors && d[i] !== 0.0) {
        for (let j = 0; j < i; j++) {
          let g = 0.0;
          for (let k = 0; k < i; k++) g += z[i * N + k] * z[k * N + j];
          for (let k = 0; k < i; k++) z[k * N + j] -= g * z[k * N + i];
        }
      }
      d[i] = z[i * N + i];
      if (computeVectors) {
        z[i * N + i] = 1.0;
        for (let j = 0; j < i; j++) {
          z[j * N + i] = 0.0;
          z[i * N + j] = 0.0;
        }
      }
    }

    for (let i = 1; i < N; i++) e[i - 1] = e[i];
    e[N - 1] = 0.0;

    for (let l = 0; l < N; l++) {
      let iter = 0;
      let m = l;
      do {
        for (m = l; m < N - 1; m++) {
          const dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
          if (Math.abs(e[m]) + dd === dd) break;
        }
        if (m !== l) {
          if (iter++ === 60) break;
          let g = (d[l + 1] - d[l]) / (2.0 * e[l]);
          let r = Math.hypot(g, 1.0);
          g = d[m] - d[l] + e[l] / (g + (g >= 0 ? Math.abs(r) : -Math.abs(r)));
          let s = 1.0;
          let c = 1.0;
          let p = 0.0;
          for (let i = m - 1; i >= l; i--) {
            const f = s * e[i];
            const b = c * e[i];
            r = Math.hypot(f, g);
            e[i + 1] = r;
            if (r === 0.0) {
              d[i + 1] -= p;
              e[m] = 0.0;
              break;
            }
            s = f / r;
            c = g / r;
            g = d[i + 1] - p;
            r = (d[i] - g) * s + 2.0 * c * b;
            p = s * r;
            d[i + 1] = g + p;
            g = c * r - b;
            if (computeVectors) {
              for (let k = 0; k < N; k++) {
                const fVec = z[k * N + i + 1];
                z[k * N + i + 1] = s * z[k * N + i] + c * fVec;
                z[k * N + i] = c * z[k * N + i] - s * fVec;
              }
            }
          }
          if (r === 0.0 && m >= l) continue;
          d[l] -= p;
          e[l] = g;
          e[m] = 0.0;
        }
      } while (m !== l);
    }

    const indices = Array.from({ length: N }, (_, idx) => idx).sort(
      (a, b) => d[a] - d[b]
    );
    const sortedVals = new Float64Array(N);
    const sortedVecs = computeVectors ? new Float64Array(N * N) : undefined;
    for (let col = 0; col < N; col++) {
      const origCol = indices[col];
      sortedVals[col] = Math.max(0.0, d[origCol]);
      if (computeVectors && sortedVecs) {
        for (let row = 0; row < N; row++) {
          sortedVecs[row * N + col] = z[row * N + origCol];
        }
      }
    }
    return { eigenvalues: sortedVals, eigenvectors: sortedVecs };
  }

  public static clusterManifoldNmeSc(
    affinityMatrix: Float64Array,
    N: number,
    minK = 1,
    maxK = 4
  ): { labels: Int32Array; optimalK: number; optimalP: number } {
    if (N <= 2 || maxK <= 1) {
      return { labels: new Int32Array(N).fill(0), optimalK: 1, optimalP: 1 };
    }

    if (minK <= 1 && this.isSingleSpeakerManifold(affinityMatrix, N)) {
      return {
        labels: new Int32Array(N).fill(0),
        optimalK: 1,
        optimalP: N - 1,
      };
    }

    const pMin = Math.max(2, Math.floor(0.06 * N));
    const pMax = Math.min(32, Math.max(pMin + 2, Math.floor(0.42 * N)));
    const candidatePs: number[] = [];
    const step = Math.max(1, Math.floor((pMax - pMin) / 14));
    for (let p = pMin; p <= pMax; p += step) candidatePs.push(p);

    let bestRatio = Infinity;
    let bestP = candidatePs[0];
    const effectiveMinK = Math.max(2, minK);
    const effectiveMaxK = Math.max(
      effectiveMinK,
      Math.min(maxK, Math.floor(N / 3))
    );
    let bestK = effectiveMinK;

    for (const p of candidatePs) {
      const laplacian = this.buildNormalizedLaplacian(affinityMatrix, N, p);
      const { eigenvalues } = this.symmetricEigenDecomposition(
        laplacian,
        N,
        false
      );

      let maxGap = -1.0;
      let argMaxK = effectiveMinK;
      for (let k = effectiveMinK; k <= effectiveMaxK; k++) {
        const gap = eigenvalues[k] - eigenvalues[k - 1];
        if (gap > maxGap) {
          maxGap = gap;
          argMaxK = k;
        }
      }
      const lambdaMax = eigenvalues[N - 1];
      const nmeGp = maxGap / (lambdaMax + 1e-10);
      const ratioRp = p / (nmeGp + 1e-10);

      if (ratioRp < bestRatio) {
        bestRatio = ratioRp;
        bestP = p;
        bestK = argMaxK;
      }
    }

    const finalLaplacian = this.buildNormalizedLaplacian(
      affinityMatrix,
      N,
      bestP
    );
    const { eigenvectors } = this.symmetricEigenDecomposition(
      finalLaplacian,
      N,
      true
    );
    const spectralEmbeddings: Float64Array[] = new Array(N);

    for (let i = 0; i < N; i++) {
      const rowVec = new Float64Array(bestK);
      let normSq = 0.0;
      for (let k = 0; k < bestK; k++) {
        const val = eigenvectors![i * N + k];
        rowVec[k] = val;
        normSq += val * val;
      }
      const invNorm = 1.0 / Math.sqrt(normSq + 1e-12);
      for (let k = 0; k < bestK; k++) rowVec[k] *= invNorm;
      spectralEmbeddings[i] = rowVec;
    }

    const labels = this.sphericalKMeansPlusPlus(spectralEmbeddings, N, bestK);
    return { labels, optimalK: bestK, optimalP: bestP };
  }

  private static isSingleSpeakerManifold(A: Float64Array, N: number): boolean {
    if (N < 4) return true;
    const offDiag: number[] = [];
    let totalBimodalGap = 0.0;

    for (let i = 0; i < N; i++) {
      const rowVals: number[] = [];
      for (let j = 0; j < N; j++) {
        if (i !== j) {
          rowVals.push(A[i * N + j]);
          offDiag.push(A[i * N + j]);
        }
      }
      rowVals.sort((a, b) => a - b);
      const mid = Math.floor(rowVals.length / 2);
      let cB = 0.0;
      let cW = 0.0;
      for (let m = 0; m < mid; m++) cB += rowVals[m];
      for (let m = mid; m < rowVals.length; m++) cW += rowVals[m];
      cB /= Math.max(1, mid);
      cW /= Math.max(1, rowVals.length - mid);
      totalBimodalGap += cW - cB;
    }

    offDiag.sort((a, b) => a - b);
    const q25 = offDiag[Math.floor(offDiag.length * 0.25)];
    const avgBimodalGap = totalBimodalGap / N;
    return q25 >= 0.48 && avgBimodalGap < 0.16;
  }

  private static buildNormalizedLaplacian(
    A: Float64Array,
    N: number,
    p: number
  ): Float64Array {
    const pruned = new Float64Array(N * N);
    const rowCopy = new Float64Array(N);

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) rowCopy[j] = A[i * N + j];
      rowCopy.sort();
      const threshold = rowCopy[Math.max(0, N - p)];
      for (let j = 0; j < N; j++) {
        pruned[i * N + j] = A[i * N + j] >= threshold ? 1.0 : 0.0;
      }
    }

    const W = new Float64Array(N * N);
    const invSqrtDeg = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let deg = 0.0;
      for (let j = 0; j < N; j++) {
        const w = 0.5 * (pruned[i * N + j] + pruned[j * N + i]);
        W[i * N + j] = w;
        deg += w;
      }
      invSqrtDeg[i] = 1.0 / Math.sqrt(deg + 1e-10);
    }

    const L = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const normW = W[i * N + j] * invSqrtDeg[i] * invSqrtDeg[j];
        L[i * N + j] = (i === j ? 1.0 : 0.0) - normW;
      }
    }
    return L;
  }

  private static sphericalKMeansPlusPlus(
    points: Float64Array[],
    N: number,
    K: number
  ): Int32Array {
    const dim = points[0].length;
    let bestLabels = new Int32Array(N);
    let bestInertia = Infinity;

    for (let trial = 0; trial < 6; trial++) {
      const centroids: Float64Array[] = [];
      const firstIdx = (trial * 17) % N;
      centroids.push(new Float64Array(points[firstIdx]));

      const distSq = new Float64Array(N).fill(Infinity);
      for (let c = 1; c < K; c++) {
        let maxD = -1.0;
        let bestIdx = 0;
        for (let i = 0; i < N; i++) {
          let dot = 0.0;
          for (let d = 0; d < dim; d++) {
            dot += points[i][d] * centroids[c - 1][d];
          }
          const d2 = Math.max(0.0, 2.0 - 2.0 * dot);
          if (d2 < distSq[i]) distSq[i] = d2;
          if (distSq[i] > maxD) {
            maxD = distSq[i];
            bestIdx = i;
          }
        }
        centroids.push(new Float64Array(points[bestIdx]));
      }

      const labels = new Int32Array(N);
      for (let iter = 0; iter < 30; iter++) {
        let changed = false;
        for (let i = 0; i < N; i++) {
          let bestC = 0;
          let bestSim = -Infinity;
          for (let c = 0; c < K; c++) {
            let sim = 0.0;
            for (let d = 0; d < dim; d++) {
              sim += points[i][d] * centroids[c][d];
            }
            if (sim > bestSim) {
              bestSim = sim;
              bestC = c;
            }
          }
          if (labels[i] !== bestC) {
            labels[i] = bestC;
            changed = true;
          }
        }
        if (!changed) break;

        for (let c = 0; c < K; c++) {
          centroids[c].fill(0.0);
          let count = 0;
          for (let i = 0; i < N; i++) {
            if (labels[i] === c) {
              for (let d = 0; d < dim; d++) {
                centroids[c][d] += points[i][d];
              }
              count++;
            }
          }
          if (count > 0) {
            let norm = 0.0;
            for (let d = 0; d < dim; d++) {
              norm += centroids[c][d] * centroids[c][d];
            }
            const inv = 1.0 / Math.sqrt(norm + 1e-12);
            for (let d = 0; d < dim; d++) centroids[c][d] *= inv;
          }
        }
      }

      let inertia = 0.0;
      for (let i = 0; i < N; i++) {
        const c = labels[i];
        let dot = 0.0;
        for (let d = 0; d < dim; d++) dot += points[i][d] * centroids[c][d];
        inertia += Math.max(0.0, 2.0 - 2.0 * dot);
      }

      if (inertia < bestInertia) {
        bestInertia = inertia;
        bestLabels = labels;
      }
    }

    return bestLabels;
  }
}

// ============================================================================
// 3. WeSpeaker ONNX (ResNet34-LM) 載入與快取推理引擎
// ============================================================================

class WeSpeakerOnnxRunner {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private session: any = null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private ortModule: any = null;
  private checked = false;
  public embeddingDim = 64;

  constructor(private config: DiarizationConfig) {}

  public async init(): Promise<boolean> {
    if (this.session) return true;
    if (this.checked) return false;
    this.checked = true;

    try {
      const candidateFiles = [
        this.config.onnxModelFilename || 'wespeaker_voxceleb_resnet34_int8.onnx',
        'wespeaker_voxceleb_resnet34_int8.onnx',
        'voxceleb_resnet34_LM.onnx',
        'nemo_en_titanet_small.int8.onnx',
      ];

      let modelBuffer: ArrayBuffer | null = null;
      let loadedFileName = '';

      for (const file of candidateFiles) {
        const url = resolvePublicRootUrl(`models/${file}`);
        try {
          const head = await fetch(url, { method: 'HEAD', cache: 'no-cache' });
          const cType = head.headers.get('content-type') || '';
          if (!head.ok || cType.includes('text/html')) continue;

          if ('caches' in self) {
            const cache = await caches.open(LOCAL_MODEL_CACHE_NAME);
            const cached = await cache.match(url);
            if (cached) {
              modelBuffer = await cached.arrayBuffer();
            } else {
              const res = await fetch(url, { cache: 'no-cache' });
              if (res.ok) {
                await cache.put(url, res.clone());
                modelBuffer = await res.arrayBuffer();
              }
            }
          } else {
            const res = await fetch(url);
            if (res.ok) modelBuffer = await res.arrayBuffer();
          }

          if (modelBuffer && modelBuffer.byteLength > 500000) {
            loadedFileName = file;
            break;
          }
        } catch {
          // next
        }
      }

      if (!modelBuffer || modelBuffer.byteLength < 500000) {
        console.info(
          'ℹ️ [Diarization] 採用極速 64維共振峰保留型 Kaldi Fbank TSTP 引擎'
        );
        return false;
      }

      this.ortModule = await import('onnxruntime-web');
      const wasmRootUrl = resolvePublicRootUrl('wasm/');
      if (this.ortModule.env?.wasm) {
        this.ortModule.env.wasm.wasmPaths = wasmRootUrl;
        this.ortModule.env.wasm.simd = true;
        this.ortModule.env.wasm.numThreads = Math.max(
          1,
          Math.min(4, Math.floor((navigator.hardwareConcurrency || 4) / 2))
        );
      }

      this.session = await this.ortModule.InferenceSession.create(modelBuffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      this.embeddingDim = 256;
      console.info(
        `✅ [WeSpeaker ONNX] 成功載入本地模型 (${loadedFileName}, ${(
          modelBuffer.byteLength /
          (1024 * 1024)
        ).toFixed(1)} MB)！`
      );
      return true;
    } catch (e) {
      console.warn('⚠️ [WeSpeaker ONNX] 切換至極速 TSTP 引擎:', e);
      return false;
    }
  }

  public async computeEmbedding(
    cmnTensor: Float32Array,
    binMeans: Float32Array,
    numFrames: number,
    fbankHelper: KaldiFbankExtractor
  ): Promise<Float32Array> {
    if (!this.session || !this.ortModule) {
      this.embeddingDim = 64;
      return fbankHelper.computeFormantPreservingTstpEmbedding(
        cmnTensor,
        binMeans,
        numFrames
      );
    }

    try {
      const ort = this.ortModule;
      const inputNames: string[] = this.session.inputNames || ['feats'];
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const feeds: Record<string, any> = {};

      if (inputNames.includes('audio_signal')) {
        const transposed = new Float32Array(80 * numFrames);
        for (let t = 0; t < numFrames; t++) {
          for (let b = 0; b < 80; b++) {
            transposed[b * numFrames + t] = cmnTensor[t * 80 + b];
          }
        }
        feeds['audio_signal'] = new ort.Tensor('float32', transposed, [
          1,
          80,
          numFrames,
        ]);
        if (inputNames.includes('length')) {
          feeds['length'] = new ort.Tensor(
            'int64',
            BigInt64Array.from([BigInt(numFrames)]),
            [1]
          );
        }
      } else {
        feeds[inputNames[0]] = new ort.Tensor('float32', cmnTensor, [
          1,
          numFrames,
          80,
        ]);
      }

      const outputMap = await this.session.run(feeds);
      const outName = this.session.outputNames.includes('embs')
        ? 'embs'
        : this.session.outputNames[0];
      const rawEmb = outputMap[outName]?.data as Float32Array | undefined;

      if (rawEmb && rawEmb.length >= 64) {
        this.embeddingDim = rawEmb.length;
        return l2Normalize(rawEmb);
      }
    } catch {
      // fallback
    }

    this.embeddingDim = 64;
    return fbankHelper.computeFormantPreservingTstpEmbedding(
      cmnTensor,
      binMeans,
      numFrames
    );
  }
}

// ============================================================================
// 4. 主執行緒入口：極速微觀 SCD 切分 + 主幹段 ONNX/TSTP 提取 + 雙層流形 NME-SC
// ============================================================================

interface HomogeneousSegmentNode {
  startSec: number;
  endSec: number;
  centerSec: number;
  startFrame: number;
  endFrame: number;
  f0Hz: number;
  scdLeft: number;
  scdRight: number;
  embedding: Float32Array;
}

function runManifoldCluster(
  nodes: HomogeneousSegmentNode[],
  minManifoldK: number,
  maxManifoldK: number
): { labels: Int32Array; k: number } {
  const N = nodes.length;
  if (N === 0) return { labels: new Int32Array(0), k: 0 };
  if (N <= 2 || maxManifoldK <= 1) {
    return { labels: new Int32Array(N).fill(0), k: 1 };
  }

  const dim = nodes[0].embedding.length;
  const meanVec = new Float32Array(dim);
  for (const n of nodes) {
    for (let d = 0; d < dim; d++) meanVec[d] += n.embedding[d];
  }
  for (let d = 0; d < dim; d++) meanVec[d] /= N;

  const centeredEmbeddings: Float32Array[] = nodes.map((n) => {
    const c = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      c[d] = n.embedding[d] - meanVec[d] * 0.55;
    }
    return l2Normalize(c);
  });

  const A = new Float64Array(N * N);
  const betaTime = 0.08;
  const sigmaTimeSq2 = 2.0 * 1.2 * 1.2;

  for (let i = 0; i < N; i++) {
    A[i * N + i] = 1.0;
    for (let j = i + 1; j < N; j++) {
      const cos = cosineSimilarity(centeredEmbeddings[i], centeredEmbeddings[j]);
      const dt = nodes[i].centerSec - nodes[j].centerSec;
      const timeKernel = betaTime * Math.exp(-(dt * dt) / sigmaTimeSq2);
      // 嚴格對應報告步驟 1 公式：A_ij = max(0, S_ij + \beta_time * exp(-\Delta t^2 / 2\sigma_t^2))
      const sim = Math.max(0.0, Math.min(1.0, cos + timeKernel));
      A[i * N + j] = sim;
      A[j * N + i] = sim;
    }
  }

  const res = AutoTuningSpectralClusterer.clusterManifoldNmeSc(
    A,
    N,
    minManifoldK,
    maxManifoldK
  );
  return { labels: res.labels, k: res.optimalK };
}

self.onmessage = async (event: MessageEvent) => {
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
  const maxSpeakers = Math.max(
    minSpeakers,
    Math.min(10, Number(data.maxSpeakers) || 6)
  );
  const existingSpeakers: Speaker[] = Array.isArray(data.existingSpeakers)
    ? data.existingSpeakers
    : [];

  const diarConfig: DiarizationConfig = {
    engineMode: data.diarizationConfig?.engineMode || 'mode_b_advanced_dsp',
    clusterMethod: data.diarizationConfig?.clusterMethod || 'spectral_cosine',
    executionBackend: data.diarizationConfig?.executionBackend || 'wasm',
    onnxModelFilename:
      data.diarizationConfig?.onnxModelFilename ||
      'wespeaker_voxceleb_resnet34_int8.onnx',
    localWasmBasePath: data.diarizationConfig?.localWasmBasePath || './wasm/',
    localModelBasePath: data.diarizationConfig?.localModelBasePath || './models/',
    yieldEveryNUnits: data.diarizationConfig?.yieldEveryNUnits ?? 14,
    enableGlobalMeanCentering:
      data.diarizationConfig?.enableGlobalMeanCentering ?? true,
    languageHint: data.diarizationConfig?.languageHint || 'zh',
    shortTurnConfidenceKeepThreshold:
      data.diarizationConfig?.shortTurnConfidenceKeepThreshold ?? 0.68,
    pingPongMaxDurSec: data.diarizationConfig?.pingPongMaxDurSec ?? 1.8,
    enablePauseAndSemanticWeighting:
      data.diarizationConfig?.enablePauseAndSemanticWeighting ?? true,
  };

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
    const fbankExtractor = new KaldiFbankExtractor();
    const onnxRunner = new WeSpeakerOnnxRunner(diarConfig);
    await onnxRunner.init();

    self.postMessage({
      type: 'DIARIZE_PROGRESS',
      progress: 12,
      detailText: '計算 Bit-Exact Kaldi 80-dim Fbank 與微觀 SCD 跳變曲線...',
    });

    const { frames: allFbankFrames, frameRms, numFrames } =
      fbankExtractor.computeGlobalFbank(samples);

    if (numFrames < 40) {
      self.postMessage({
        type: 'DIARIZE_SUCCESS',
        turns: [],
        asrWindows: [],
        updatedSpeakers: existingSpeakers,
      });
      return;
    }

    const sortedRms = Array.from(frameRms).sort((a, b) => a - b);
    const noiseFloor = sortedRms[Math.floor(sortedRms.length * 0.18)] || 0.004;
    const vadThreshold = Math.max(0.008, noiseFloor * 2.0);

    // ── Step 1: 極速計算 Scale-3 (0.8s 視窗, 0.2s 步長) 微觀聲紋跳變曲線 d_SCD(t) (僅需 ~15ms) ──
    const scdCurveF = new Float32Array(numFrames);
    for (let f = 80; f <= numFrames - 80; f += 20) {
      if (frameRms[f] < vadThreshold * 0.8) continue;
      const leftSlice = fbankExtractor.sliceAndApplyCMN(allFbankFrames, f - 80, f);
      const rightSlice = fbankExtractor.sliceAndApplyCMN(allFbankFrames, f, f + 80);
      const embL = fbankExtractor.computeFormantPreservingTstpEmbedding(
        leftSlice.cmnTensor,
        leftSlice.binMeans,
        80
      );
      const embR = fbankExtractor.computeFormantPreservingTstpEmbedding(
        rightSlice.cmnTensor,
        rightSlice.binMeans,
        80
      );
      const scdVal = Math.max(0.0, 1.0 - cosineSimilarity(embL, embR));
      for (let k = Math.max(0, f - 10); k < Math.min(numFrames, f + 10); k++) {
        scdCurveF[k] = scdVal;
      }
    }

    // ── Step 2: 結合 VAD 靜音與 SCD 波峰切出約 80~140 個同質語音段 (Homogeneous Sub-segments) ──
    const rawSubBounds: Array<{ startF: number; endF: number }> = [];
    let inSpeech = false;
    let startF = 0;
    let silenceFrames = 0;

    for (let f = 0; f < numFrames; f++) {
      if (frameRms[f] >= vadThreshold) {
        if (!inSpeech) {
          inSpeech = true;
          startF = f;
        }
        silenceFrames = 0;

        const curLenF = f - startF;
        // 若連續發聲超過 1.1s 且遇到顯著 SCD 換人波峰 (或超過 3.2s)，切分為獨立子段
        const isScdPeak =
          curLenF >= 110 &&
          scdCurveF[f] >= 0.24 &&
          scdCurveF[f] >= (scdCurveF[Math.max(0, f - 20)] || 0) &&
          scdCurveF[f] >= (scdCurveF[Math.min(numFrames - 1, f + 20)] || 0);

        if (isScdPeak || curLenF >= 320) {
          rawSubBounds.push({ startF, endF: f });
          startF = f;
        }
      } else if (inSpeech) {
        silenceFrames++;
        if (silenceFrames >= 25) {
          const endF = f - silenceFrames + 1;
          if (endF - startF >= 28) {
            rawSubBounds.push({ startF, endF });
          }
          inSpeech = false;
          silenceFrames = 0;
        }
      }
    }
    if (inSpeech && numFrames - startF >= 28) {
      rawSubBounds.push({ startF, endF: numFrames });
    }

    // ── Step 3: 僅對這 ~100 個純淨語音段執行多尺度 (2.4s 宏觀 + 本身中觀) 聲紋提取 ──
    const segNodes: HomogeneousSegmentNode[] = [];

    for (let idx = 0; idx < rawSubBounds.length; idx++) {
      const b = rawSubBounds[idx];
      const centerF = Math.floor((b.startF + b.endF) / 2);

      // Scale 1 (2.4s 宏觀上下文) 與 Scale 2 (當前語音段本身)
      const s1Start = Math.max(b.startF, centerF - 120);
      const s1End = Math.min(b.endF, centerF + 120);

      const sSample = Math.max(0, b.startF * 160);
      const eSample = Math.min(samples.length, b.endF * 160 + 400);
      const f0Hz = estimateMedianPitchHz(samples, sSample, eSample, sampleRate);

      const sliceSeg = fbankExtractor.sliceAndApplyCMN(
        allFbankFrames,
        b.startF,
        b.endF
      );
      const embSeg = await onnxRunner.computeEmbedding(
        sliceSeg.cmnTensor,
        sliceSeg.binMeans,
        b.endF - b.startF,
        fbankExtractor
      );

      let finalEmb = embSeg;
      if (b.endF - b.startF > 180) {
        const sliceMacro = fbankExtractor.sliceAndApplyCMN(
          allFbankFrames,
          s1Start,
          s1End
        );
        const embMacro = await onnxRunner.computeEmbedding(
          sliceMacro.cmnTensor,
          sliceMacro.binMeans,
          s1End - s1Start,
          fbankExtractor
        );
        const fused = new Float32Array(embSeg.length);
        for (let d = 0; d < embSeg.length; d++) {
          fused[d] = 0.65 * embMacro[d] + 0.35 * embSeg[d];
        }
        finalEmb = l2Normalize(fused);
      }

      segNodes.push({
        startSec: chunkStartSec + b.startF * 0.01,
        endSec: chunkStartSec + b.endF * 0.01,
        centerSec: chunkStartSec + centerF * 0.01,
        startFrame: b.startF,
        endFrame: b.endF,
        f0Hz,
        scdLeft: Number((scdCurveF[b.startF] || 0).toFixed(3)),
        scdRight: Number((scdCurveF[Math.min(numFrames - 1, b.endF)] || 0).toFixed(3)),
        embedding: finalEmb,
      });

      if ((idx + 1) % 10 === 0 || idx === rawSubBounds.length - 1) {
        const pct = 15 + Math.round(((idx + 1) / rawSubBounds.length) * 70);
        self.postMessage({
          type: 'DIARIZE_PROGRESS',
          progress: pct,
          detailText: `提取語音段深度聲紋 (${idx + 1}/${rawSubBounds.length} 段)...`,
        });
        await new Promise((r) => setTimeout(r, 2));
      }
    }

    if (segNodes.length === 0) {
      self.postMessage({
        type: 'DIARIZE_SUCCESS',
        turns: [],
        asrWindows: [],
        updatedSpeakers: existingSpeakers,
      });
      return;
    }

    const embDim = segNodes[0].embedding.length;

    // ── Step 4: 雙層流形獨立 NME-SC 聚類 (徹底消除 K=2 性別鴻溝陷阱) ──
    self.postMessage({
      type: 'DIARIZE_PROGRESS',
      progress: 90,
      detailText: '執行雙層流形 Auto-Tuning NME-SC 譜聚類...',
    });

    const highManifoldNodes: HomogeneousSegmentNode[] = [];
    const highOrigIndices: number[] = [];
    const lowManifoldNodes: HomogeneousSegmentNode[] = [];
    const lowOrigIndices: number[] = [];

    segNodes.forEach((node, idx) => {
      if (node.f0Hz >= 162) {
        highManifoldNodes.push(node);
        highOrigIndices.push(idx);
      } else {
        lowManifoldNodes.push(node);
        lowOrigIndices.push(idx);
      }
    });

    const N_obs = segNodes.length;
    const allLabels = new Int32Array(N_obs);
    let totalClusters = 0;

    const hasBothManifolds =
      highManifoldNodes.length >= 3 && lowManifoldNodes.length >= 3;

    if (hasBothManifolds) {
      const highDur = highManifoldNodes.reduce(
        (acc, n) => acc + (n.endSec - n.startSec),
        0
      );
      const lowDur = lowManifoldNodes.reduce(
        (acc, n) => acc + (n.endSec - n.startSec),
        0
      );
      const minHighK = highDur >= 10 && highManifoldNodes.length >= 4 ? 2 : 1;
      const minLowK = lowDur >= 10 && lowManifoldNodes.length >= 4 ? 2 : 1;
      const maxHighK = Math.max(minHighK, Math.min(3, maxSpeakers - minLowK));
      const maxLowK = Math.max(minLowK, Math.min(3, maxSpeakers - minHighK));

      const highRes = runManifoldCluster(highManifoldNodes, minHighK, maxHighK);
      const lowRes = runManifoldCluster(lowManifoldNodes, minLowK, maxLowK);

      for (let i = 0; i < highManifoldNodes.length; i++) {
        allLabels[highOrigIndices[i]] = highRes.labels[i];
      }
      for (let i = 0; i < lowManifoldNodes.length; i++) {
        allLabels[lowOrigIndices[i]] = highRes.k + lowRes.labels[i];
      }
      totalClusters = highRes.k + lowRes.k;
    } else {
      const singleRes = runManifoldCluster(segNodes, minSpeakers, maxSpeakers);
      for (let i = 0; i < N_obs; i++) {
        allLabels[i] = singleRes.labels[i];
      }
      totalClusters = singleRes.k;
    }

    interface LocalClusterSummary {
      clusterIdx: number;
      obsIndices: number[];
      centroid: Float32Array;
      exemplars: Float32Array[];
      f0Hz: number;
      totalDurSec: number;
    }

    const validLocalClusters: LocalClusterSummary[] = [];

    for (let k = 0; k < totalClusters; k++) {
      const obsIdxList: number[] = [];
      for (let i = 0; i < N_obs; i++) {
        if (allLabels[i] === k) obsIdxList.push(i);
      }
      if (obsIdxList.length === 0) continue;

      const meanVec = new Float32Array(embDim);
      let weightedF0 = 0.0;
      let durSum = 0.0;

      for (const oIdx of obsIdxList) {
        const node = segNodes[oIdx];
        const w = Math.max(0.3, node.endSec - node.startSec);
        durSum += w;
        weightedF0 += node.f0Hz * w;
        for (let d = 0; d < embDim; d++) {
          meanVec[d] += node.embedding[d] * w;
        }
      }

      const normCentroid = l2Normalize(meanVec);
      const scoredNodes = obsIdxList
        .map((idx) => ({
          idx,
          sim: cosineSimilarity(segNodes[idx].embedding, normCentroid),
        }))
        .sort((a, b) => b.sim - a.sim);

      const exemplars: Float32Array[] = [];
      const highPurityPool = scoredNodes
        .slice(0, Math.max(1, Math.ceil(scoredNodes.length * 0.6)))
        .map((item) => segNodes[item.idx].embedding);

      if (highPurityPool.length > 0) {
        exemplars.push(highPurityPool[0]);
        while (exemplars.length < 6 && exemplars.length < highPurityPool.length) {
          let bestCand = highPurityPool[0];
          let minMaxSim = Infinity;
          for (const cand of highPurityPool) {
            let maxSimToEx = -1.0;
            for (const ex of exemplars) {
              const s = cosineSimilarity(cand, ex);
              if (s > maxSimToEx) maxSimToEx = s;
            }
            if (maxSimToEx < minMaxSim) {
              minMaxSim = maxSimToEx;
              bestCand = cand;
            }
          }
          if (minMaxSim > 0.98) break;
          exemplars.push(bestCand);
        }
      }

      validLocalClusters.push({
        clusterIdx: k,
        obsIndices: obsIdxList,
        centroid: normCentroid,
        exemplars,
        f0Hz: durSum > 0 ? weightedF0 / durSum : 155,
        totalDurSec: durSum,
      });
    }

    const totalAllDur =
      validLocalClusters.reduce((acc, c) => acc + c.totalDurSec, 0) || 1;
    while (validLocalClusters.length > minSpeakers) {
      const tinyIdx = validLocalClusters.findIndex(
        (c) => c.totalDurSec / totalAllDur < 0.045
      );
      if (tinyIdx === -1) break;

      let bestTarget = -1;
      let bestSim = -Infinity;
      for (let i = 0; i < validLocalClusters.length; i++) {
        if (i === tinyIdx) continue;
        if (
          (validLocalClusters[i].f0Hz >= 162) !==
          (validLocalClusters[tinyIdx].f0Hz >= 162)
        ) {
          continue;
        }
        const s = cosineSimilarity(
          validLocalClusters[tinyIdx].centroid,
          validLocalClusters[i].centroid
        );
        if (s > bestSim) {
          bestSim = s;
          bestTarget = i;
        }
      }
      if (bestTarget === -1) break;
      validLocalClusters[bestTarget].obsIndices.push(
        ...validLocalClusters[tinyIdx].obsIndices
      );
      validLocalClusters[bestTarget].totalDurSec +=
        validLocalClusters[tinyIdx].totalDurSec;
      validLocalClusters.splice(tinyIdx, 1);
    }

    validLocalClusters.sort((a, b) => b.totalDurSec - a.totalDurSec);
    if (validLocalClusters.length > maxSpeakers) {
      validLocalClusters.length = maxSpeakers;
    }

    const updatedSpeakers: Speaker[] = existingSpeakers.map((sp) => ({
      ...sp,
      centroid: sp.centroid ? [...sp.centroid] : undefined,
      embedding: sp.embedding ? [...sp.embedding] : undefined,
      anchorBank: sp.anchorBank ? sp.anchorBank.map((a) => [...a]) : undefined,
      mergedFrom: Array.isArray(sp.mergedFrom) ? [...sp.mergedFrom] : [],
    }));

    const usedGlobalSpeakerIds = new Set<string>();
    const activeProfiles = new Map<
      string,
      { f0Hz: number; masterCentroid: Float32Array; anchorBank: Float32Array[] }
    >();

    for (const loc of validLocalClusters) {
      let bestSp: Speaker | null = null;
      let bestOmega = -1.0;

      for (const sp of updatedSpeakers) {
        if (usedGlobalSpeakerIds.has(sp.id)) continue;
        const spF0 = sp.f0Hz ?? sp.centroid?.[0] ?? 155;
        if ((loc.f0Hz >= 162) !== (spF0 >= 162)) continue;

        const spEmb =
          sp.embedding && sp.embedding.length === embDim ? sp.embedding : null;
        if (!spEmb) continue;

        const masterSim = cosineSimilarity(loc.centroid, spEmb);
        let maxAnchorSim = masterSim;
        if (Array.isArray(sp.anchorBank)) {
          for (const anc of sp.anchorBank) {
            if (anc.length === embDim) {
              const s = cosineSimilarity(loc.centroid, anc);
              if (s > maxAnchorSim) maxAnchorSim = s;
            }
          }
        }
        const omega = 0.65 * masterSim + 0.35 * maxAnchorSim;
        if (omega > bestOmega) {
          bestOmega = omega;
          bestSp = sp;
        }
      }

      if (bestSp && bestOmega >= 0.45) {
        usedGlobalSpeakerIds.add(bestSp.id);
        if (bestOmega >= 0.56) {
          const oldMass = bestSp.accumulatedMass || 10.0;
          const alpha = Math.min(
            0.25,
            loc.totalDurSec / (oldMass + loc.totalDurSec)
          );
          const oldEmb = bestSp.embedding!;
          const blended = new Float32Array(embDim);
          for (let d = 0; d < embDim; d++) {
            blended[d] = (1.0 - alpha) * oldEmb[d] + alpha * loc.centroid[d];
          }
          const normMaster = l2Normalize(blended);
          const newF0 =
            (bestSp.f0Hz ?? loc.f0Hz) * (1.0 - alpha) + loc.f0Hz * alpha;
          bestSp.f0Hz = newF0;
          bestSp.embedding = Array.from(normMaster);
          bestSp.centroid = [newF0, ...Array.from(normMaster)];
          bestSp.accumulatedMass = oldMass + loc.totalDurSec;
          bestSp.anchorBank = [
            ...(bestSp.anchorBank || []).slice(0, 3),
            ...loc.exemplars.slice(0, 3).map((e) => Array.from(e)),
          ].slice(0, 6);
        }

        activeProfiles.set(bestSp.id, {
          f0Hz: bestSp.f0Hz ?? loc.f0Hz,
          masterCentroid: l2Normalize(bestSp.embedding!),
          anchorBank: (bestSp.anchorBank || []).map((a) => l2Normalize(a)),
        });
      } else if (updatedSpeakers.length < maxSpeakers) {
        const nextNum = updatedSpeakers.length + 1;
        const newId = `Speaker_${String(nextNum).padStart(2, '0')}`;
        const genderHint = loc.f0Hz >= 162 ? '女聲' : '男聲';
        const newSp: Speaker = {
          id: newId,
          name: `語者 ${String(nextNum).padStart(2, '0')} (${genderHint})`,
          color: SPEAKER_PALETTE[(nextNum - 1) % SPEAKER_PALETTE.length],
          f0Hz: loc.f0Hz,
          embedding: Array.from(loc.centroid),
          anchorBank: loc.exemplars.map((e) => Array.from(e)),
          accumulatedMass: loc.totalDurSec,
          centroid: [loc.f0Hz, ...Array.from(loc.centroid)],
          mergedFrom: [],
        };
        updatedSpeakers.push(newSp);
        usedGlobalSpeakerIds.add(newId);
        activeProfiles.set(newId, {
          f0Hz: loc.f0Hz,
          masterCentroid: loc.centroid,
          anchorBank: loc.exemplars,
        });
      }
    }

    const fineTimelineTurns: FineDiarizationTurn[] = [];

    for (let i = 0; i < segNodes.length; i++) {
      const node = segNodes[i];
      const nodeIsHigh = node.f0Hz >= 162;

      let top1Sp = updatedSpeakers[0]?.id || 'Speaker_01';
      let top2Sp = top1Sp;
      let s1 = -2.0;
      let s2 = -2.0;

      for (const [spId, prof] of activeProfiles.entries()) {
        const profIsHigh = prof.f0Hz >= 162;
        if (profIsHigh !== nodeIsHigh) continue;

        const cSim = cosineSimilarity(node.embedding, prof.masterCentroid);
        let maxASim = cSim;
        for (const anc of prof.anchorBank) {
          const s = cosineSimilarity(node.embedding, anc);
          if (s > maxASim) maxASim = s;
        }
        const compScore = 0.6 * cSim + 0.4 * maxASim;

        if (compScore > s1) {
          s2 = s1;
          top2Sp = top1Sp;
          s1 = compScore;
          top1Sp = spId;
        } else if (compScore > s2) {
          s2 = compScore;
          top2Sp = spId;
        }
      }

      const margin = Math.max(0.0, s1 - (s2 > -1.5 ? s2 : 0.0));

      fineTimelineTurns.push({
        start: Number(node.startSec.toFixed(2)),
        end: Number(node.endSec.toFixed(2)),
        speakerId: top1Sp,
        runnerUpSpeakerId: top2Sp,
        f0Hz: Math.round(node.f0Hz),
        speakerConfidence: Number(
          Math.max(0.5, Math.min(0.99, (s1 + 1.0) / 2.0)).toFixed(3)
        ),
        speakerMargin: Number(margin.toFixed(3)),
        boundaryScdLeft: node.scdLeft,
        boundaryScdRight: node.scdRight,
        isProtectedBackchannel:
          margin >= 0.14 ||
          (margin >= 0.06 && node.scdLeft + node.scdRight >= 0.45),
      });
    }

    for (let i = 1; i < fineTimelineTurns.length - 1; i++) {
      const prev = fineTimelineTurns[i - 1];
      const curr = fineTimelineTurns[i];
      const next = fineTimelineTurns[i + 1];
      const dur = curr.end - curr.start;

      if (
        prev.speakerId === next.speakerId &&
        curr.speakerId !== prev.speakerId &&
        dur < 1.5
      ) {
        const marginBA = curr.speakerMargin || 0.0;
        const dualScdPeak =
          (curr.boundaryScdLeft || 0) + (curr.boundaryScdRight || 0);
        const isTrueBackchannel =
          marginBA >= 0.14 || (marginBA >= 0.06 && dualScdPeak >= 0.45);

        if (!isTrueBackchannel) {
          curr.speakerId = prev.speakerId;
        } else {
          curr.isProtectedBackchannel = true;
        }
      }
    }

    const asrWindows: Array<{ start: number; end: number }> = [];
    let winStart = fineTimelineTurns[0].start;
    let lastEnd = fineTimelineTurns[0].end;

    for (let i = 1; i < fineTimelineTurns.length; i++) {
      const cur = fineTimelineTurns[i];
      const gap = cur.start - lastEnd;
      const currentWinDur = lastEnd - winStart;

      if (
        (currentWinDur >= 18.0 && gap >= 0.3) ||
        currentWinDur >= 25.0 ||
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

    console.info(
      `🎯 [NME-SC 完成] 共識別 ${updatedSpeakers.length} 位語者，切出 ${fineTimelineTurns.length} 個聲紋區間`
    );

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