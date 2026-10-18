// src/services/modelCacheManager.ts
import {
  GranularityTier,
  HardwareProbeResult,
  TierPresetSpec,
} from '../types';

/** 三段式硬體規格與校對細膩度常數查表 */
export const TIER_PRESET_SPECS: Record<GranularityTier, TierPresetSpec> = {
  eco_safe: {
    tier: 'eco_safe',
    label: '輕量穩定（標準文書機）',
    badgeText: 'VRAM < 2.2GB',
    hardwareHint: '適用 16GB RAM + 整合式內顯 (Intel Iris Xe / UHD / AMD 內顯)',
    chunkDurationSec: 300,
    keepModelsCoResident: false,
    recommendedLlm: 'gemma-2-2b-it-q4f16_1-MLC',
    asrBeamSize: 1,
    highAttentionChars: 300,
    mediumAttentionChars: 1200,
    maxRelistenQuotaPerChunk: 2,
    relistenPaddingSec: 0.6,
  },
  balanced_pro: {
    tier: 'balanced_pro',
    label: '進階平衡（高階筆電 / 雲端協同）',
    badgeText: 'VRAM ~ 2.6GB',
    hardwareHint: '適用 32GB RAM 筆電、4~6GB 獨顯，或已設定 Gemini API Key 之電腦',
    chunkDurationSec: 300,
    keepModelsCoResident: false,
    recommendedLlm: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    asrBeamSize: 2,
    highAttentionChars: 300,
    mediumAttentionChars: 1500,
    maxRelistenQuotaPerChunk: 5,
    relistenPaddingSec: 0.8,
  },
  ultra_studio: {
    tier: 'ultra_studio',
    label: '旗艦精修（獨顯工作站）',
    badgeText: '雙模型常駐 ~ 4.2GB',
    hardwareHint: '適用 32GB+ RAM 搭配 8GB+ VRAM 獨立顯卡 (RTX 3060 / 4060 以上)',
    chunkDurationSec: 300,
    keepModelsCoResident: true,
    recommendedLlm: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    asrBeamSize: 3,
    highAttentionChars: 400,
    mediumAttentionChars: 2400,
    maxRelistenQuotaPerChunk: 10,
    relistenPaddingSec: 1.2,
  },
};

/**
 * 探測瀏覽器 WebGPU 適配器規格、單一 Buffer 上限與 IndexedDB 快取佔用
 */
export async function probeHardwareAndCache(): Promise<HardwareProbeResult> {
  let estimatedCacheUsageMB = 0;
  if (navigator.storage && navigator.storage.estimate) {
    try {
      const estimate = await navigator.storage.estimate();
      estimatedCacheUsageMB = Math.round((estimate.usage || 0) / (1024 * 1024));
    } catch {
      estimatedCacheUsageMB = 0;
    }
  }

  // 修正 TS2304: 將型別明確指定為 unknown 避免找不到 GPU 型別定義
  const gpu = (navigator as unknown as { gpu?: unknown }).gpu as
    | {
        requestAdapter(options?: { powerPreference?: string }): Promise<{
          limits?: { maxBufferSize?: number; maxStorageBufferBindingSize?: number };
        } | null>;
      }
    | undefined;

  if (!gpu) {
    return {
      hasWebGPU: false,
      adapterName: '不支援 WebGPU (僅限雲端 BYOK 模式)',
      isDiscreteGpu: false,
      maxBufferSizeMB: 0,
      estimatedCacheUsageMB,
      recommendedTier: 'eco_safe',
      warningMessage: '目前瀏覽器未啟用 WebGPU，請使用 Chrome 113+ 或切換至雲端模式。',
    };
  }

  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      return {
        hasWebGPU: false,
        adapterName: '找不到可用之 GPU 適配器',
        isDiscreteGpu: false,
        maxBufferSizeMB: 0,
        estimatedCacheUsageMB,
        recommendedTier: 'eco_safe',
      };
    }

    const info = (adapter as unknown as { info?: { vendor?: string; architecture?: string; description?: string } }).info || {};
    const vendor = (info.vendor || '').toLowerCase();
    const desc = (info.description || info.architecture || 'WebGPU Adapter').toLowerCase();

    const maxBufferBytes = adapter.limits?.maxBufferSize || 268435456;
    const maxStorageBytes = adapter.limits?.maxStorageBufferBindingSize || 134217728;
    const maxBufferSizeMB = Math.round(Math.max(maxBufferBytes, maxStorageBytes) / (1024 * 1024));

    const isIntegratedKeyword =
      desc.includes('iris') ||
      desc.includes('uhd') ||
      desc.includes('hd graphics') ||
      (vendor.includes('intel') && !desc.includes('arc'));

    const isDiscreteKeyword =
      vendor.includes('nvidia') ||
      desc.includes('rtx') ||
      desc.includes('gtx') ||
      desc.includes('radeon rx') ||
      desc.includes('arc');

    const isDiscreteGpu = isDiscreteKeyword || (!isIntegratedKeyword && maxBufferSizeMB >= 2048);

    let recommendedTier: GranularityTier = 'eco_safe';
    if (isDiscreteGpu && maxBufferSizeMB >= 2048) {
      recommendedTier = 'ultra_studio';
    } else if (maxBufferSizeMB >= 1024 && !isIntegratedKeyword) {
      recommendedTier = 'balanced_pro';
    }

    const displayAdapterName = info.description || `${info.vendor || 'Generic'} GPU (${maxBufferSizeMB}MB Buffer)`;

    return {
      hasWebGPU: true,
      adapterName: displayAdapterName,
      isDiscreteGpu,
      maxBufferSizeMB,
      estimatedCacheUsageMB,
      recommendedTier,
    };
  } catch (err) {
    return {
      hasWebGPU: false,
      adapterName: `GPU 探測異常: ${(err as Error).message}`,
      isDiscreteGpu: false,
      maxBufferSizeMB: 0,
      estimatedCacheUsageMB,
      recommendedTier: 'eco_safe',
    };
  }
}

export function validateTierAgainstHardware(
  selectedTier: GranularityTier,
  probe: HardwareProbeResult | null,
  hasCloudApiKey: boolean
): string | null {
  if (!probe) return null;
  if (selectedTier === 'ultra_studio' && !probe.isDiscreteGpu && !hasCloudApiKey) {
    return '偵測到您的電腦為整合式內顯，於純離線模式開啟「旗艦精修」極易記憶體不足，建議改用「輕量穩定」或填入 Gemini API Key。';
  }
  return null;
}