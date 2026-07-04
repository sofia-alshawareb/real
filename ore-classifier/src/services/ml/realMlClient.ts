import type { Frame } from '../../types/models';
import type { SegmentationResult } from '../ml/types';
import { encodeUiMaskIndexPngBase64 } from '../maskExport';
import { mlApiUrl } from './config';
import { MlUnavailableError } from './errors';
import { getFrameImageBlob } from './imagePayload';
import type { CalibClassKey } from './classMap';
import { decodePalettedPngBase64, encodeBinaryHintMaskPngBase64, prepareMaskForFrame, upscaleMaskNearest } from './maskUtils';

const POLL_INTERVAL_MS = 1500;
const JOB_TIMEOUT_MS = 5 * 60 * 1000;

interface UploadResponse {
  image_id: string;
  width: number;
  height: number;
}

interface JobResponse {
  job_id: string;
  image_id: string;
  status: 'pending' | 'running' | 'done' | 'failed';
  progress: number;
  error: string | null;
}

interface MaskResponse {
  image_id: string;
  mask_width: number;
  mask_height: number;
  native_width: number;
  native_height: number;
  mask_to_native_scale: number;
  encoding: string;
  data: string;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch {
    throw new MlUnavailableError('Не удалось связаться с сервером ML-анализа');
  }
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new MlUnavailableError(
      detail ? `Сервис ML: ${detail}` : `Сервис ML вернул ошибку ${res.status}`,
    );
  }
  return res.json() as Promise<T>;
}

export async function uploadFrameImage(frame: Frame): Promise<string> {
  if (frame.backendImageId) return frame.backendImageId;
  const blob = await getFrameImageBlob(frame);
  const form = new FormData();
  form.append('file', blob, `${frame.name || frame.id}.png`);
  const result = await fetchJson<UploadResponse>(mlApiUrl('/api/v1/images'), {
    method: 'POST',
    body: form,
  });
  return result.image_id;
}

async function pollJob(jobId: string): Promise<void> {
  const deadline = Date.now() + JOB_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const job = await fetchJson<JobResponse>(mlApiUrl(`/api/v1/jobs/${jobId}`));
    if (job.status === 'done') return;
    if (job.status === 'failed') {
      throw new MlUnavailableError(job.error ?? 'Сегментация на сервере завершилась с ошибкой');
    }
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
  throw new MlUnavailableError('Превышено время ожидания сегментации на сервере');
}

export async function segmentFrameReal(
  frame: Frame,
  onProgress?: (share: number) => void,
): Promise<SegmentationResult & { backendImageId: string }> {
  onProgress?.(0.05);
  const backendImageId = await uploadFrameImage(frame);
  onProgress?.(0.15);

  const job = await fetchJson<JobResponse>(mlApiUrl('/api/v1/jobs/segment'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: backendImageId, save_activations: false }),
  });

  onProgress?.(0.25);
  await pollJob(job.job_id);
  onProgress?.(0.85);

  const mask = await fetchJson<MaskResponse>(mlApiUrl(`/api/v1/images/${backendImageId}/mask`));
  const { labels, width, height } = await decodePalettedPngBase64(mask.data);
  const prepared = prepareMaskForFrame(labels, width, height, frame.width, frame.height);
  onProgress?.(1);

  return {
    ...prepared,
    confidence: 0.9,
    backendImageId,
  };
}

interface RefineDefectResponse extends MaskResponse {
  refinement?: unknown;
}

export async function refineCalibrationFromHint(
  backendImageId: string,
  hintMaskNative: Uint8Array,
  nativeWidth: number,
  nativeHeight: number,
  frameNativeW: number,
  frameNativeH: number,
  uiClass: CalibClassKey,
): Promise<{ data: Uint8Array; mw: number; mh: number }> {
  const hintB64 = await encodeBinaryHintMaskPngBase64(hintMaskNative, nativeWidth, nativeHeight);
  const resp = await fetchJson<RefineDefectResponse>(
    mlApiUrl(`/api/v1/images/${backendImageId}/refine/calibration`),
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hint_mask: hintB64, ui_class: uiClass }),
    },
  );
  const { labels, width, height } = await decodePalettedPngBase64(resp.data);
  return prepareMaskForFrame(labels, width, height, frameNativeW, frameNativeH);
}

/** @deprecated Use refineCalibrationFromHint */
export async function refineDefectFromHint(
  backendImageId: string,
  hintMaskNative: Uint8Array,
  nativeWidth: number,
  nativeHeight: number,
  frameNativeW: number,
  frameNativeH: number,
): Promise<{ data: Uint8Array; mw: number; mh: number }> {
  return refineCalibrationFromHint(
    backendImageId,
    hintMaskNative,
    nativeWidth,
    nativeHeight,
    frameNativeW,
    frameNativeH,
    'talc',
  );
}

interface SaveManualMaskResponse {
  image_id: string;
  artifact_path: string;
  width: number;
  height: number;
  files: string[];
}

/** Upload hand-drawn UI mask to server artifact store (dev). */
export async function saveUserDrawnMaskToServer(
  frame: Frame,
  labelsWorking: Uint8Array,
  workingWidth: number,
  workingHeight: number,
): Promise<SaveManualMaskResponse> {
  const backendImageId = await uploadFrameImage(frame);
  const nativeLabels =
    workingWidth === frame.width && workingHeight === frame.height
      ? labelsWorking
      : upscaleMaskNearest(labelsWorking, workingWidth, workingHeight, frame.width, frame.height);
  const maskB64 = await encodeUiMaskIndexPngBase64(nativeLabels, frame.width, frame.height);
  return fetchJson<SaveManualMaskResponse>(mlApiUrl(`/api/v1/images/${backendImageId}/manual-mask`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mask: maskB64 }),
  });
}
