import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import OpenSeadragon from 'openseadragon';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import CircularProgress from '@mui/material/CircularProgress';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import SaveIcon from '@mui/icons-material/Save';
import { useHotkeys } from 'react-hotkeys-hook';
import { useExperimentsStore } from '../../stores/experimentsStore';
import { useDepositsStore } from '../../stores/depositsStore';
import { useEditorStore, type MaskClassKey } from '../../stores/editorStore';
import { MASK_CLASSES } from '../../theme/palette';
import { maskWorkingSize } from '../../services/grainModel';
import { getMask, putMask } from '../../db/imageRepo';
import { calcMetrics } from '../../services/metricsCalc';
import { classifyFrame } from '../../services/rulesEngine';
import { ViewerCanvas } from './ViewerCanvas';
import { MaskOverlay, type MaskBuffer } from './maskOverlay';
import { stampSegment, rasterizePolygon, imagePointToMask, type Point2D } from './maskPainter';
import { useUndoRedo } from './useUndoRedo';
import { EditorToolbar } from './EditorToolbar';
import { LayerLegend } from './LayerLegend';
import { FrameClassPanel } from './FrameClassPanel';
import { ConfirmDialog } from '../../components/ConfirmDialog';
import { genId } from '../../stores/experimentsStore';
import { notify } from '../../utils/toast';

const SAVE_DEBOUNCE_MS = 2000;

export function FrameEditorPage() {
  const { experimentId, frameId } = useParams<{ experimentId: string; frameId: string }>();
  const navigate = useNavigate();

  const experiment = useExperimentsStore((s) => s.experiments.find((e) => e.id === experimentId));
  const updateFrame = useExperimentsStore((s) => s.updateFrame);
  const setFrameResult = useExperimentsStore((s) => s.setFrameResult);
  const setManualClassOverride = useExperimentsStore((s) => s.setManualClassOverride);
  const deposit = useDepositsStore((s) => s.deposits.find((d) => d.id === experiment?.depositId));

  const frame = experiment?.frames.find((f) => f.id === frameId);

  const tool = useEditorStore((s) => s.tool);
  const setTool = useEditorStore((s) => s.setTool);
  const activeClass = useEditorStore((s) => s.activeClass);
  const setActiveClass = useEditorStore((s) => s.setActiveClass);
  const brushRadius = useEditorStore((s) => s.brushRadius);
  const setBrushRadius = useEditorStore((s) => s.setBrushRadius);
  const overlayOpacity = useEditorStore((s) => s.overlayOpacity);
  const setOverlayOpacity = useEditorStore((s) => s.setOverlayOpacity);
  const visibleLayers = useEditorStore((s) => s.visibleLayers);
  const toggleLayer = useEditorStore((s) => s.toggleLayer);
  const compareMode = useEditorStore((s) => s.compareMode);
  const setCompareMode = useEditorStore((s) => s.setCompareMode);

  const undoRedo = useUndoRedo();

  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const overlayRef = useRef<MaskOverlay | null>(null);
  const maskBufferRef = useRef<MaskBuffer | null>(null);
  const maskToNativeScaleRef = useRef(1);
  const editedMaskDataRef = useRef<Uint8Array | null>(null);
  const isStrokingRef = useRef(false);
  const lastPointRef = useRef<Point2D | null>(null);
  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [polygonPoints, setPolygonPoints] = useState<Point2D[]>([]);

  const [maskLoading, setMaskLoading] = useState(true);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved'>('idle');
  const [revertConfirmOpen, setRevertConfirmOpen] = useState(false);
  const [viewerVersion, setViewerVersion] = useState(0);

  const talcThreshold = deposit?.talcThreshold ?? 0.1;

  // Загрузка/создание маски при смене кадра
  useEffect(() => {
    let cancelled = false;
    setMaskLoading(true);
    undoRedo.clear();
    setPolygonPoints([]);

    async function load() {
      if (!frame) return;
      let record = frame.maskId ? await getMask(frame.maskId) : undefined;
      if (!record) {
        const { mw, mh, scale } = maskWorkingSize(frame.width, frame.height);
        const data = new Uint8Array(mw * mh);
        const newId = genId('mask');
        record = { id: newId, frameId: frame.id, width: mw, height: mh, data };
        await putMask(record);
        maskToNativeScaleRef.current = scale;
        updateFrame(experimentId!, frame.id, { maskId: newId, status: frame.status === 'queued' || frame.status === 'ml_unavailable' ? 'manual_only' : frame.status });
      } else {
        const { scale } = maskWorkingSize(frame.width, frame.height);
        maskToNativeScaleRef.current = scale;
      }
      if (cancelled) return;
      maskBufferRef.current = { width: record.width, height: record.height, data: record.data };
      editedMaskDataRef.current = record.data;
      setMaskLoading(false);
      setViewerVersion((v) => v + 1);
    }
    void load();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame?.id]);

  const performSave = useCallback(async () => {
    if (!maskBufferRef.current || !frame || !experimentId) return;
    setSaveStatus('saving');
    const buffer = maskBufferRef.current;
    await putMask({ id: frame.maskId!, frameId: frame.id, width: buffer.width, height: buffer.height, data: buffer.data });
    const metrics = calcMetrics(buffer, frame.pixelSizeUm, maskToNativeScaleRef.current);
    const { oreClass, reason } = classifyFrame(metrics, talcThreshold);
    updateFrame(experimentId, frame.id, { manuallyEditedMask: true });
    setFrameResult(experimentId, frame.id, {
      status: 'segmentation_edited',
      maskId: frame.maskId!,
      metrics,
      frameClass: oreClass,
      classReason: reason,
      confidence: frame.confidence,
    });
    setSaveStatus('saved');
    notify('Сохранено, метрики пересчитаны', 'success');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame, experimentId, talcThreshold]);

  const scheduleSave = useCallback(() => {
    setSaveStatus('idle');
    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    saveTimeoutRef.current = setTimeout(() => void performSave(), SAVE_DEBOUNCE_MS);
  }, [performSave]);

  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
        void performSave();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame?.id]);

  // Создание оверлея после готовности viewer + маски
  const handleViewerReady = useCallback((viewer: OpenSeadragon.Viewer) => {
    viewerRef.current = viewer;
    setViewerVersion((v) => v + 1);
  }, []);

  useEffect(() => {
    if (!viewerRef.current || !maskBufferRef.current || maskLoading) return;
    overlayRef.current?.destroy();
    const overlay = new MaskOverlay(viewerRef.current, maskBufferRef.current, maskToNativeScaleRef.current);
    overlay.setVisibleLayers(visibleLayers);
    overlay.setOpacity(overlayOpacity);
    overlayRef.current = overlay;
    return () => {
      overlay.destroy();
      if (overlayRef.current === overlay) overlayRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewerVersion, maskLoading]);

  useEffect(() => {
    overlayRef.current?.setVisibleLayers(visibleLayers);
  }, [visibleLayers]);
  useEffect(() => {
    overlayRef.current?.setOpacity(overlayOpacity);
  }, [overlayOpacity]);

  // Compare mode: подменяем данные оверлея на автомаску
  useEffect(() => {
    if (!overlayRef.current || !frame) return;
    if (compareMode && frame.autoMaskId) {
      void getMask(frame.autoMaskId).then((rec) => {
        if (rec && overlayRef.current) overlayRef.current.setMask({ width: rec.width, height: rec.height, data: rec.data });
      });
    } else if (maskBufferRef.current) {
      overlayRef.current.setMask(maskBufferRef.current);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compareMode]);

  const pointerToMask = useCallback((e: React.PointerEvent): Point2D | null => {
    const overlayCanvas = overlayRef.current?.getOverlayCanvas();
    const viewer = viewerRef.current;
    if (!overlayCanvas || !viewer) return null;
    const rect = overlayCanvas.getBoundingClientRect();
    const localX = e.clientX - rect.left;
    const localY = e.clientY - rect.top;
    const imgPoint = viewer.viewport.viewerElementToImageCoordinates(new OpenSeadragon.Point(localX, localY));
    return imagePointToMask(imgPoint.x, imgPoint.y, maskToNativeScaleRef.current);
  }, []);

  const paintingEnabled = !compareMode && (tool === 'brush' || tool === 'eraser' || tool === 'polygon');

  const handlePointerDown = (e: React.PointerEvent) => {
    if (!paintingEnabled || !maskBufferRef.current) return;
    const p = pointerToMask(e);
    if (!p) return;
    if (tool === 'polygon') {
      setPolygonPoints((prev) => [...prev, p]);
      return;
    }
    undoRedo.snapshotBeforeEdit(maskBufferRef.current.data);
    const value = tool === 'eraser' ? 0 : MASK_CLASSES[activeClass].value;
    const rect = stampSegment(maskBufferRef.current, p.x, p.y, p.x, p.y, brushRadius, value);
    overlayRef.current?.updateRegion(rect.x, rect.y, rect.w, rect.h);
    isStrokingRef.current = true;
    lastPointRef.current = p;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isStrokingRef.current || !maskBufferRef.current || !lastPointRef.current) return;
    const p = pointerToMask(e);
    if (!p) return;
    const value = tool === 'eraser' ? 0 : MASK_CLASSES[activeClass].value;
    const rect = stampSegment(maskBufferRef.current, lastPointRef.current.x, lastPointRef.current.y, p.x, p.y, brushRadius, value);
    overlayRef.current?.updateRegion(rect.x, rect.y, rect.w, rect.h);
    lastPointRef.current = p;
  };

  const handlePointerUp = () => {
    if (!isStrokingRef.current) return;
    isStrokingRef.current = false;
    lastPointRef.current = null;
    scheduleSave();
  };

  const closePolygon = useCallback(() => {
    if (!maskBufferRef.current || polygonPoints.length < 3) {
      setPolygonPoints([]);
      return;
    }
    undoRedo.snapshotBeforeEdit(maskBufferRef.current.data);
    const value = MASK_CLASSES[activeClass].value;
    const rect = rasterizePolygon(maskBufferRef.current, polygonPoints, value);
    overlayRef.current?.updateRegion(rect.x, rect.y, rect.w, rect.h);
    setPolygonPoints([]);
    scheduleSave();
  }, [polygonPoints, activeClass, undoRedo, scheduleSave]);

  const handleUndo = useCallback(() => {
    if (!maskBufferRef.current) return;
    const prev = undoRedo.undo(maskBufferRef.current.data);
    if (!prev) return;
    maskBufferRef.current.data = prev;
    overlayRef.current?.setMask(maskBufferRef.current);
    scheduleSave();
  }, [undoRedo, scheduleSave]);

  const handleRedo = useCallback(() => {
    if (!maskBufferRef.current) return;
    const next = undoRedo.redo(maskBufferRef.current.data);
    if (!next) return;
    maskBufferRef.current.data = next;
    overlayRef.current?.setMask(maskBufferRef.current);
    scheduleSave();
  }, [undoRedo, scheduleSave]);

  const handleRevertToAuto = useCallback(async () => {
    if (!frame?.autoMaskId || !maskBufferRef.current) return;
    const autoRec = await getMask(frame.autoMaskId);
    if (!autoRec) return;
    undoRedo.snapshotBeforeEdit(maskBufferRef.current.data);
    maskBufferRef.current.data = autoRec.data.slice();
    overlayRef.current?.setMask(maskBufferRef.current);
    setRevertConfirmOpen(false);
    void performSave();
  }, [frame, undoRedo, performSave]);

  const frameIndex = experiment?.frames.findIndex((f) => f.id === frameId) ?? -1;
  const goToFrame = useCallback(
    (delta: number) => {
      if (!experiment) return;
      const nextIndex = frameIndex + delta;
      const next = experiment.frames[nextIndex];
      if (next) navigate(`/experiments/${experiment.id}/frames/${next.id}`);
    },
    [experiment, frameIndex, navigate],
  );

  useHotkeys('1', () => setActiveClass('sulfide'), []);
  useHotkeys('2', () => setActiveClass('gangue'), []);
  useHotkeys('3', () => setActiveClass('talc'), []);
  useHotkeys('b', () => setTool('brush'), []);
  useHotkeys('e', () => setTool('eraser'), []);
  useHotkeys('p', () => setTool('polygon'), []);
  useHotkeys('v', () => setTool('pan'), []);
  useHotkeys('[', () => setBrushRadius(Math.max(2, brushRadius - 4)), [brushRadius]);
  useHotkeys(']', () => setBrushRadius(Math.min(128, brushRadius + 4)), [brushRadius]);
  useHotkeys('ctrl+z', handleUndo, [handleUndo]);
  useHotkeys('ctrl+y', handleRedo, [handleRedo]);
  useHotkeys('enter', closePolygon, [closePolygon]);
  useHotkeys('escape', () => setPolygonPoints([]), []);
  useHotkeys('left', () => goToFrame(-1), [goToFrame]);
  useHotkeys('right', () => goToFrame(1), [goToFrame]);

  if (!experiment || !frame) {
    return (
      <Box>
        <Typography>Кадр не найден.</Typography>
        <Button startIcon={<ArrowBackIcon />} onClick={() => navigate(-1)}>
          Назад
        </Button>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        height: "calc(100vh - 88px)"
      }}>
      <Stack
        direction="row"
        spacing={1}
        sx={{
          alignItems: "center",
          mb: 1.5
        }}>
        <IconButton onClick={() => navigate(`/experiments/${experiment.id}`)}>
          <ArrowBackIcon />
        </IconButton>
        <Typography variant="h6" noWrap sx={{
          flexGrow: 1
        }}>
          {experiment.title} · {frame.name}
        </Typography>
        <IconButton disabled={frameIndex <= 0} onClick={() => goToFrame(-1)}>
          <ArrowBackIosNewIcon fontSize="small" />
        </IconButton>
        <Typography variant="body2" sx={{
          color: "text.secondary"
        }}>
          {frameIndex + 1} / {experiment.frames.length}
        </Typography>
        <IconButton disabled={frameIndex >= experiment.frames.length - 1} onClick={() => goToFrame(1)}>
          <ArrowForwardIosIcon fontSize="small" />
        </IconButton>
        {polygonPoints.length > 0 && (
          <Chip label={`Полигон: ${polygonPoints.length} точек · Enter — замкнуть, Esc — отмена`} color="info" size="small" />
        )}
        {compareMode && <Chip label="Режим сравнения с авторазметкой" color="secondary" size="small" />}
        <Chip
          icon={saveStatus === 'saving' ? <CircularProgress size={14} /> : <SaveIcon fontSize="small" />}
          label={saveStatus === 'saving' ? 'Сохранение...' : saveStatus === 'saved' ? 'Сохранено' : 'Без изменений'}
          size="small"
          variant="outlined"
        />
      </Stack>
      <Stack
        direction="row"
        spacing={2}
        sx={{
          flexGrow: 1,
          minHeight: 0
        }}>
        <EditorToolbar
          tool={tool}
          onToolChange={setTool}
          activeClass={activeClass}
          onActiveClassChange={setActiveClass}
          brushRadius={brushRadius}
          onBrushRadiusChange={setBrushRadius}
          overlayOpacity={overlayOpacity}
          onOverlayOpacityChange={setOverlayOpacity}
          canUndo={undoRedo.canUndo}
          canRedo={undoRedo.canRedo}
          onUndo={handleUndo}
          onRedo={handleRedo}
          onRevertToAuto={() => setRevertConfirmOpen(true)}
          hasAutoMask={Boolean(frame.autoMaskId)}
          compareMode={compareMode}
          onToggleCompare={() => setCompareMode(!compareMode)}
          disabled={maskLoading}
        />

        <Box
          sx={{
            flexGrow: 1,
            position: "relative",
            minWidth: 0
          }}>
          <ViewerCanvas frame={frame} onViewerReady={handleViewerReady} />
          {(maskLoading || !overlayRef.current) && (
            <Box
              sx={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                bgcolor: 'rgba(32,36,42,0.6)',
              }}
            >
              <CircularProgress sx={{ color: '#fff' }} />
            </Box>
          )}
          <Box
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onDoubleClick={() => tool === 'polygon' && closePolygon()}
            style={{ pointerEvents: paintingEnabled ? 'auto' : 'none' }}
            sx={{
              position: 'absolute',
              inset: 0,
              cursor: paintingEnabled ? 'crosshair' : 'default',
            }}
          />
        </Box>

        <Stack
          spacing={2}
          sx={{
            width: 280,
            flexShrink: 0,
            overflowY: 'auto'
          }}>
          <LayerLegend visibleLayers={visibleLayers} onToggle={(k: MaskClassKey) => toggleLayer(k)} metrics={frame.metrics} />
          <FrameClassPanel
            frame={frame}
            depositId={experiment.depositId}
            onConfirm={() => updateFrame(experiment.id, frame.id, { status: 'reviewed' })}
            onManualOverride={(oreClass) => setManualClassOverride(experiment.id, frame.id, oreClass, experiment.author)}
          />
        </Stack>
      </Stack>
      <ConfirmDialog
        open={revertConfirmOpen}
        title="Вернуть автосегментацию?"
        description="Все ручные правки текущего кадра будут потеряны и заменены результатом автоматической обработки."
        confirmLabel="Вернуть"
        danger
        onConfirm={handleRevertToAuto}
        onCancel={() => setRevertConfirmOpen(false)}
      />
    </Box>
  );
}
