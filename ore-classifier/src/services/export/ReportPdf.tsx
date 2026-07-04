import { Document, Page, Text, View, StyleSheet, Font, Image as PdfImage } from '@react-pdf/renderer';
import type { Deposit, Experiment, OreClass, ReportDraft } from '../../types/models';
import { ORE_CLASS_META } from '../../theme/palette';
import { ROLE_LABELS } from '../../features/deposits/MineralProfileEditor';
import { formatDateTime } from '../../utils/format';

Font.register({
  family: 'PT Sans',
  fonts: [
    { src: '/fonts/PTSans-Regular.ttf', fontWeight: 'normal' },
    { src: '/fonts/PTSans-Bold.ttf', fontWeight: 'bold' },
  ],
});
Font.registerHyphenationCallback((word) => [word]);

const DERIVATION_LABEL: Record<string, string> = {
  majority: 'по большинству кадров',
  reference: 'по опорному кадру',
  manual: 'вручную',
};

const styles = StyleSheet.create({
  page: { padding: 32, fontFamily: 'PT Sans', fontSize: 10, color: '#20242A' },
  h1: { fontSize: 18, fontWeight: 'bold', marginBottom: 4 },
  meta: { fontSize: 9, color: '#666', marginBottom: 12 },
  sectionTitle: { fontSize: 13, fontWeight: 'bold', marginTop: 16, marginBottom: 6 },
  paragraph: { lineHeight: 1.4 },
  row: { flexDirection: 'row', borderBottomWidth: 1, borderBottomColor: '#eeeeee', paddingVertical: 4 },
  headerRow: { flexDirection: 'row', borderBottomWidth: 1, borderBottomColor: '#333', paddingVertical: 4 },
  cell: { flex: 1, fontSize: 9 },
  headerCell: { flex: 1, fontSize: 9, fontWeight: 'bold' },
  swatch: { width: 8, height: 8, borderRadius: 4, marginRight: 4 },
  mineralRow: { flexDirection: 'row', alignItems: 'center', borderBottomWidth: 1, borderBottomColor: '#eeeeee', paddingVertical: 4 },
  thumbGrid: { marginTop: 8 },
  thumb: { marginBottom: 12 },
  thumbPairRow: { flexDirection: 'row', gap: 8, marginTop: 4 },
  thumbImg: { width: 150, height: 95, objectFit: 'cover', marginBottom: 2, borderRadius: 4 },
  thumbCaption: { fontSize: 7, color: '#777' },
  thumbLabel: { fontSize: 8, color: '#555', marginTop: 2 },
});

export interface FrameThumbnailPair {
  original: string;
  masked: string;
}

interface ReportPdfProps {
  experiment: Experiment;
  deposit?: Deposit;
  draft: ReportDraft;
  frameThumbnails: Record<string, FrameThumbnailPair>;
}

function classLabel(cls?: OreClass): string {
  return cls ? ORE_CLASS_META[cls].label : '—';
}

export function ReportPdf({ experiment, deposit, draft, frameThumbnails }: ReportPdfProps) {
  const includedFrames = experiment.frames.filter((f) => draft.includedFrameIds.includes(f.id));
  const depositMetaLine = deposit
    ? [deposit.oreCluster, deposit.region, deposit.oreTypes?.length ? deposit.oreTypes.join(', ') : undefined]
        .filter(Boolean)
        .join(' · ')
    : '';

  return (
    <Document>
      <Page size="A4" style={styles.page}>
        <Text style={styles.h1}>{experiment.title}</Text>
        <Text style={styles.meta}>
          Месторождение: {deposit?.name ?? '—'} · Автор: {experiment.author} · Дата формирования: {formatDateTime(Date.now())}
        </Text>
        {depositMetaLine && <Text style={styles.meta}>{depositMetaLine}</Text>}

        <Text style={styles.sectionTitle}>Сводка</Text>
        <Text style={styles.paragraph}>Итоговый класс руды: {classLabel(experiment.experimentClass)}</Text>
        <Text style={styles.paragraph}>
          Способ определения: {DERIVATION_LABEL[experiment.classDerivation ?? 'majority']}
        </Text>
        <Text style={styles.paragraph}>
          Режим анализа: {{ ml: 'автоматический (ML)', manual: 'ручной', mixed: 'смешанный' }[experiment.analysisMode]}
        </Text>

        <Text style={styles.sectionTitle}>Введение</Text>
        <Text style={styles.paragraph}>{draft.intro || '—'}</Text>

        <Text style={styles.sectionTitle}>Количественные метрики по кадрам</Text>
        <View style={styles.headerRow}>
          <Text style={styles.headerCell}>Кадр</Text>
          <Text style={styles.headerCell}>Класс</Text>
          <Text style={styles.headerCell}>Сульфиды всего</Text>
          <Text style={styles.headerCell}>Обычные</Text>
          <Text style={styles.headerCell}>Тонкие</Text>
          <Text style={styles.headerCell}>Тальк</Text>
        </View>
        {includedFrames.map((f) => (
          <View style={styles.row} key={f.id} wrap={false}>
            <Text style={styles.cell}>{f.name}</Text>
            <Text style={styles.cell}>{classLabel(f.manualClassOverride ?? f.frameClass)}</Text>
            <Text style={styles.cell}>{f.metrics ? `${(f.metrics.sulfideFraction * 100).toFixed(1)}%` : '—'}</Text>
            <Text style={styles.cell}>{f.metrics ? `${(f.metrics.coarseFraction * 100).toFixed(1)}%` : '—'}</Text>
            <Text style={styles.cell}>{f.metrics ? `${(f.metrics.fineFraction * 100).toFixed(1)}%` : '—'}</Text>
            <Text style={styles.cell}>{f.metrics ? `${(f.metrics.talcFraction * 100).toFixed(1)}%` : '—'}</Text>
          </View>
        ))}

        {deposit && deposit.minerals.length > 0 && (
          <>
            <Text style={styles.sectionTitle}>Профиль минералов месторождения</Text>
            <View style={styles.headerRow}>
              <Text style={styles.headerCell}>Минерал</Text>
              <Text style={styles.headerCell}>Роль</Text>
              <Text style={[styles.headerCell, { flex: 2 }]}>Заметка</Text>
            </View>
            {deposit.minerals.map((m) => (
              <View style={styles.mineralRow} key={m.id} wrap={false}>
                <View style={[styles.swatch, { backgroundColor: m.colorHex }]} />
                <Text style={styles.cell}>{m.name}</Text>
                <Text style={styles.cell}>{ROLE_LABELS[m.role]}</Text>
                <Text style={[styles.cell, { flex: 2 }]}>{draft.mineralNotes?.[m.id] ?? m.note ?? '—'}</Text>
              </View>
            ))}
          </>
        )}

        <Text style={styles.sectionTitle}>Иллюстрации (исходник / маска)</Text>
        <View style={styles.thumbGrid}>
          {includedFrames.map(
            (f) =>
              frameThumbnails[f.id] && (
                <View style={styles.thumb} key={f.id} wrap={false}>
                  <Text style={styles.thumbLabel}>{f.name}</Text>
                  <View style={styles.thumbPairRow}>
                    <View>
                      <PdfImage src={frameThumbnails[f.id].original} style={styles.thumbImg} />
                      <Text style={styles.thumbCaption}>Исходное изображение</Text>
                    </View>
                    <View>
                      <PdfImage src={frameThumbnails[f.id].masked} style={styles.thumbImg} />
                      <Text style={styles.thumbCaption}>С маской сегментации</Text>
                    </View>
                  </View>
                </View>
              ),
          )}
        </View>

        <Text style={styles.sectionTitle}>Выводы</Text>
        <Text style={styles.paragraph}>{draft.conclusion || '—'}</Text>

        <Text style={styles.sectionTitle}>Рекомендации</Text>
        <Text style={styles.paragraph}>{draft.recommendations || '—'}</Text>
      </Page>
    </Document>
  );
}
