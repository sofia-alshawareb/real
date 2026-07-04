import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams, useBlocker } from 'react-router-dom';
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
import type { FrameMetrics } from '../../types/models';
import { ViewerCanvas } from './ViewerCanvas';
import { MaskOverlay, type MaskBuffer } from './maskOverlay';
import { stampSegment, rasterizePolygon, imagePointToMask, floodFill, type Point2D } from './maskPainter';
import { useUndoRedo } from './useUndoRedo';
import { EditorToolbar } from './EditorToolbar';
import { LayerLegend } from './LayerLegend';
import { FrameClassPanel } from './FrameClassPanel';
import { ConfirmDialog } from '../../components/ConfirmDialog';
import { genId } from '../../stores/experimentsStore';
import { notify } from '../../utils/toast';

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
  const viewMode = useEditorStore((s) => s.viewMode);
  const setViewMode = useEditorStore((s) => s.setViewMode);

  const undoRedo = useUndoRedo();

  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const overlayRef = useRef<MaskOverlay | null>(null);
  const maskBufferRef = useRef<MaskBuffer | null>(null);
  const maskToNativeScaleRef = useRef(1);
  const isStrokingRef = useRef(false);
  const lastPointRef = useRef<Point2D | null>(null);
  const [polygonPoints, setPolygonPoints] = useState<Point2D[]>([]);
  const isLassoingRef = useRef(false);
  const lassoPointsRef = useRef<Point2D[]>([]);
  const lassoEraseRef = useRef(false);
  const [lassoPointCount, setLassoPointCount] = useState(0);

  const [maskLoading, setMaskLoading] = useState(true);
  const [overlayReady, setOverlayReady] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [liveMetrics, setLiveMetrics] = useState<FrameMetrics | null>(null);
  const [saving, setSaving] = useState(false);
  const [revertConfirmOpen, setRevertConfirmOpen] = useState(false);
  const [clearConfirmOpen, setClearConfirmOpen] = useState(false);
  const [viewerVersion, setViewerVersion] = useState(0);

  const dirtyRef = useRef(false);
  useEffect(() => {
    dirtyRef.current = dirty;
  }, [dirty]);

  const talcThreshold = deposit?.talcThreshold ?? 0.1;

  // Предупреждение при закрытии/обновлении вкладки с несохранёнными правками
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (dirtyRef.current) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, []);

  // Блокировка навигации внутри приложения (переход к другому кадру, назад и т.д.) при несохранённых правках
  const blocker = useBlocker(
    ({ currentLocation, nextLocation }) => dirtyRef.current && currentLocation.pathname !== nextLocation.pathname,
  );

  // Загрузка/создание маски при смене кадра
  useEffect(() => {
    let cancelled = false;
    setMaskLoading(true);
    setOverlayReady(false);
    setDirty(false);
    setLiveMetrics(null);
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
    setSaving(true);
    const buffer = maskBufferRef.current;
    await putMask({ id: frame.maskId!, frameId: frame.id, width: buffer.width, height: buffer.height, data: buffer.data });
    const metrics = calcMetrics(buffer);
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
    setSaving(false);
    setDirty(false);
    setLiveMetrics(null);
    notify('Сохранено, метрики пересчитаны', 'success');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame, experimentId, talcThreshold]);

  const markDirty = useCallback(() => {
    setDirty(true);
    if (maskBufferRef.current) setLiveMetrics(calcMetrics(maskBufferRef.current));
  }, []);

  // Разрешение конфликта навигации: сохранить и продолжить / не сохранять / отмена
  const handleBlockedSave = useCallback(async () => {
    await performSave();
    blocker.proceed?.();
  }, [performSave, blocker]);

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
    overlay.setViewMode(viewMode);
    overlayRef.current = overlay;
    setOverlayReady(true);
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
  useEffect(() => {
    overlayRef.current?.setViewMode(viewMode);
  }, [viewMode]);

  // Превью контура полигона обновляется при добавлении/сбросе точек (резиновая линия к курсору — в handlePointerMove)
  useEffect(() => {
    if (tool !== 'polygon') return;
    overlayRef.current?.setPreview(polygonPoints.length > 0 ? polygonPoints : null, false);
  }, [polygonPoints, tool]);

  // Скрываем превью при смене инструмента на не связанный с контуром
  useEffect(() => {
    if (tool !== 'polygon' && tool !== 'lasso') {
      overlayRef.current?.setPreview(null);
    }
  }, [tool]);

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

  const paintingEnabled =
    !compareMode && (tool === 'brush' || tool === 'eraser' || tool === 'polygon' || tool === 'fill' || tool === 'lasso');

  const handlePointerDown = (e: React.PointerEvent) => {
    if (!paintingEnabled || !maskBufferRef.current) return;
    const p = pointerToMask(e);
    if (!p) return;
    if (tool === 'polygon') {
      setPolygonPoints((prev) => [...prev, p]);
      return;
    }
    if (tool === 'fill') {
      undoRedo.snapshotBeforeEdit(maskBufferRef.current.data);
      const fillValue = e.altKey ? 0 : MASK_CLASSES[activeClass].value;
      const rect = floodFill(maskBufferRef.current, p.x, p.y, fillValue);
      if (rect) {
        overlayRef.current?.updateRegion(rect.x, rect.y, rect.w, rect.h);
        markDirty();
      }
      return;
    }
    if (tool === 'lasso') {
      isLassoingRef.current = true;
      lassoEraseRef.current = e.altKey;
      lassoPointsRef.current = [p];
      setLassoPointCount(1);
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
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
    if (isLassoingRef.current) {
      const p = pointerToMask(e);
      if (!p) return;
      const last = lassoPointsRef.current[lassoPointsRef.current.length - 1];
      if (!last || Math.hypot(p.x - last.x, p.y - last.y) >= 1.5) {
        lassoPointsRef.current.push(p);
        setLassoPointCount(lassoPointsRef.current.length);
      }
      overlayRef.current?.setPreview(lassoPointsRef.current, false);
      return;
    }
    if (tool === 'polygon' && paintingEnabled) {
      const p = pointerToMask(e);
      if (p) overlayRef.current?.setPreview([...polygonPoints, p], false);
      return;
    }
    if (!isStrokingRef.current || !maskBufferRef.current || !lastPointRef.current) return;
    const p = pointerToMask(e);
    if (!p) return;
    const value = tool === 'eraser' ? 0 : MASK_CLASSES[activeClass].value;
    const rect = stampSegment(maskBufferRef.current, lastPointRef.current.x, lastPointRef.current.y, p.x, p.y, brushRadius, value);
    overlayRef.current?.updateRegion(rect.x, rect.y, rect.w, rect.h);
    lastPointRef.current = p;
  };

  const handlePointerUp = () => {
    if (isLassoingRef.current) {
      isLassoingRef.current = false;
      const points = lassoPointsRef.current;
      lassoPointsRef.current = [];
      setLassoPointCount(0);
      overlayRef.current?.setPreview(null);
      if (points.length >= 3 && maskBufferRef.current) {
        undoRedo.snapshotBeforeEdit(maskBufferRef.current.data);
        const value = lassoEraseRef.current ? 0 : MASK_CLASSES[activeClass].value;
        const rect = rasterizePolygon(maskBufferRef.current, points, value);
        if (rect.w > 0 && rect.h > 0) {
          overlayRef.current?.updateRegion(rect.x, rect.y, rect.w, rect.h);
          markDirty();
        }
      }
      return;
    }
    if (!isStrokingRef.current) return;
    isStrokingRef.current = false;
    lastPointRef.current = null;
    markDirty();
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
    markDirty();
  }, [polygonPoints, activeClass, undoRedo, markDirty]);

  const handleUndo = useCallback(() => {
    if (!maskBufferRef.current) return;
    const prev = undoRedo.undo(maskBufferRef.current.data);
    if (!prev) return;
    maskBufferRef.current.data = prev;
    overlayRef.current?.setMask(maskBufferRef.current);
    markDirty();
  }, [undoRedo, markDirty]);

  const handleRedo = useCallback(() => {
    if (!maskBufferRef.current) return;
    const next = undoRedo.redo(maskBufferRef.current.data);
    if (!next) return;
    maskBufferRef.current.data = next;
    overlayRef.current?.setMask(maskBufferRef.current);
    markDirty();
  }, [undoRedo, markDirty]);

  const handleRevertToAuto = useCallback(async () => {
    if (!frame?.autoMaskId || !maskBufferRef.current) return;
    const autoRec = await getMask(frame.autoMaskId);
    if (!autoRec) return;
    undoRedo.snapshotBeforeEdit(maskBufferRef.current.data);
    maskBufferRef.current.data = autoRec.data.slice();
    overlayRef.current?.setMask(maskBufferRef.current);
    setRevertConfirmOpen(false);
    markDirty();
  }, [frame, undoRedo, markDirty]);

  const handleClearMask = useCallback(() => {
    if (!maskBufferRef.current) return;
    undoRedo.snapshotBeforeEdit(maskBufferRef.current.data);
    maskBufferRef.current.data.fill(0);
    overlayRef.current?.setMask(maskBufferRef.current);
    setClearConfirmOpen(false);
    markDirty();
  }, [undoRedo, markDirty]);

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

  useHotkeys('1', () => setActiveClass('coarse' as MaskClassKey), []);
  useHotkeys('2', () => setActiveClass('fine' as MaskClassKey), []);
  useHotkeys('3', () => setActiveClass('talc' as MaskClassKey), []);
  useHotkeys('4', () => setActiveClass('matrix' as MaskClassKey), []);
  useHotkeys('b', () => setTool('brush'), []);
  useHotkeys('e', () => setTool('eraser'), []);
  useHotkeys('p', () => setTool('polygon'), []);
  useHotkeys('g', () => setTool('fill'), []);
  useHotkeys('l', () => setTool('lasso'), []);
  useHotkeys('v', () => setTool('pan'), []);
  useHotkeys('[', () => setBrushRadius(Math.max(2, brushRadius - 4)), [brushRadius]);
  useHotkeys(']', () => setBrushRadius(Math.min(128, brushRadius + 4)), [brushRadius]);
  useHotkeys('ctrl+z', handleUndo, [handleUndo]);
  useHotkeys('ctrl+y', handleRedo, [handleRedo]);
  useHotkeys('ctrl+s', (e) => {
    e.preventDefault();
    void performSave();
  }, [performSave]);
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
        {lassoPointCount > 0 && (
          <Chip label={`Лассо: обводка (${lassoPointCount} точек) · отпустите кнопку — заливка`} color="info" size="small" />
        )}
        {compareMode && <Chip label="Режим сравнения с авторазметкой" color="secondary" size="small" />}
        {dirty && <Chip label="Есть несохранённые изменения" color="warning" size="small" variant="outlined" />}
        <Button
          variant="contained"
          size="small"
          startIcon={saving ? <CircularProgress size={14} color="inherit" /> : <SaveIcon fontSize="small" />}
          onClick={() => void performSave()}
          disabled={!dirty || saving}
        >
          {saving ? 'Сохранение...' : 'Сохранить изменения'}
        </Button>
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
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          onClearMask={() => setClearConfirmOpen(true)}
          disabled={maskLoading}
        />

        <Box
          sx={{
            flexGrow: 1,
            position: "relative",
            minWidth: 0
          }}>
          <ViewerCanvas frame={frame} onViewerReady={handleViewerReady} />
          {(maskLoading || !overlayReady) && (
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
            data-testid="painting-overlay"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onDoubleClick={() => tool === 'polygon' && closePolygon()}
            style={{ pointerEvents: paintingEnabled ? 'auto' : 'none' }}
            sx={{
              position: 'absolute',
              inset: 0,
              cursor: paintingEnabled ? (tool === 'fill' || tool === 'lasso' ? 'copy' : 'crosshair') : 'default',
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
          <LayerLegend
            visibleLayers={visibleLayers}
            onToggle={(k: MaskClassKey) => toggleLayer(k)}
            metrics={liveMetrics ?? frame.metrics}
            activeClass={activeClass}
            onActiveClassChange={setActiveClass}
          />
          <FrameClassPanel
            frame={frame}
            depositId={experiment.depositId}
            talcThreshold={talcThreshold}
            liveMetrics={liveMetrics ?? undefined}
            onConfirm={() => {
              updateFrame(experiment.id, frame.id, { status: 'reviewed' });
              notify('Класс кадра утверждён', 'success');
            }}
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
      <ConfirmDialog
        open={clearConfirmOpen}
        title="Очистить маску?"
        description="Вся разметка текущего кадра будет удалена. Действие можно отменить кнопкой «Отменить» до сохранения."
        confirmLabel="Очистить"
        danger
        onConfirm={handleClearMask}
        onCancel={() => setClearConfirmOpen(false)}
      />
      <UnsavedChangesDialog
        open={blocker.state === 'blocked'}
        onSave={() => void handleBlockedSave()}
        onDiscard={() => blocker.proceed?.()}
        onCancel={() => blocker.reset?.()}
      />
    </Box>
  );
}

function UnsavedChangesDialog({
  open,
  onSave,
  onDiscard,
  onCancel,
}: {
  open: boolean;
  onSave: () => void;
  onDiscard: () => void;
  onCancel: () => void;
}) {
  if (!open) return null;
  return (
    <Box
      sx={{
        position: 'fixed',
        inset: 0,
        bgcolor: 'rgba(0,0,0,0.4)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: (t) => t.zIndex.modal,
      }}
      onClick={onCancel}
    >
      <Box
        onClick={(e) => e.stopPropagation()}
        sx={{ bgcolor: 'background.paper', borderRadius: 2, p: 3, maxWidth: 420, boxShadow: 6 }}
      >
        <Typography variant="h6" gutterBottom>
          Сохранить изменения?
        </Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary', mb: 3 }}>
          В маске или классе кадра есть несохранённые изменения. Сохранить их перед тем, как продолжить?
        </Typography>
        <Stack direction="row" spacing={1} sx={{ justifyContent: 'flex-end' }}>
          <Button onClick={onCancel} color="inherit">
            Отмена
          </Button>
          <Button onClick={onDiscard} color="error">
            Не сохранять
          </Button>
          <Button onClick={onSave} variant="contained">
            Сохранить и продолжить
          </Button>
        </Stack>
      </Box>
    </Box>
  );
}
