// Живая проверка исправления «чёрного» исходного изображения в отчёте для загруженных (dexie) кадров.
// Запуск: node verify-report-thumb-live.mjs (dev-сервер должен быть поднят на localhost:5173)
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { chromium } from 'playwright';

const BASE_URL = 'http://localhost:5173';
const results = [];

function check(name, condition, extra = '') {
  results.push({ name, ok: Boolean(condition), extra });
  console.log(`${condition ? 'PASS' : 'FAIL'} — ${name}${extra ? ' (' + extra + ')' : ''}`);
}

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on('pageerror', (err) => console.log('PAGE ERROR:', err.message));
  page.on('console', (msg) => {
    if (msg.type() === 'error') console.log('CONSOLE ERROR:', msg.text());
  });

  // ---------- 0. Сгенерировать заведомо цветное (не чёрное) тестовое изображение большого размера ----------
  // Большой размер (> TILE_SIZE) важен, чтобы у пирамиды тайлов было больше одного уровня —
  // именно тогда воспроизводится баг с рендером уровня 0.
  const dataUrl = await page.evaluate(async () => {
    const canvas = document.createElement('canvas');
    canvas.width = 2000;
    canvas.height = 1200;
    const ctx = canvas.getContext('2d');
    const grad = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    grad.addColorStop(0, '#e53935');
    grad.addColorStop(0.5, '#43a047');
    grad.addColorStop(1, '#1e88e5');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < 40; i++) {
      ctx.beginPath();
      ctx.fillStyle = i % 2 === 0 ? '#ffee58' : '#fafafa';
      ctx.arc(Math.random() * canvas.width, Math.random() * canvas.height, 40 + Math.random() * 60, 0, Math.PI * 2);
      ctx.fill();
    }
    return canvas.toDataURL('image/png');
  });
  const base64 = dataUrl.split(',')[1];
  const tmpPath = path.join(os.tmpdir(), `verify-report-thumb-${Date.now()}.png`);
  fs.writeFileSync(tmpPath, Buffer.from(base64, 'base64'));

  // ---------- 1. Создать эксперимент с загруженным (не процедурным) изображением ----------
  await page.goto(`${BASE_URL}/experiments/new`, { waitUntil: 'networkidle' });
  await page.getByLabel('Название эксперимента').fill('Проверка миниатюры отчёта');
  await page.getByLabel('Месторождение').click();
  await page.locator('.MuiAutocomplete-option').first().click();

  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles(tmpPath);
  await page.waitForFunction(
    () => !document.body.innerText.includes('Обработка изображений...'),
    { timeout: 20000 },
  );
  await page.waitForTimeout(300);

  const createButton = page.getByRole('button', { name: 'Создать эксперимент' });
  check('Кнопка создания эксперимента доступна после импорта изображения', !(await createButton.isDisabled()));
  await createButton.click();
  await page.waitForURL(/\/experiments\/[^/]+$/, { timeout: 10000 }).catch(() => {});
  await page.waitForTimeout(1000);
  check('Эксперимент создан, произошёл переход на карточку эксперимента', /\/experiments\/[^/]+$/.test(page.url()));

  // ---------- 2. Открыть отчёт и проверить, что "Исходное изображение" не чёрное/однородное ----------
  await page.getByRole('button', { name: 'Сформировать отчёт' }).click();
  await page.waitForTimeout(1500);

  const originalCaption = page.locator('text=Исходное изображение').first();
  check('В отчёте есть подпись "Исходное изображение"', (await originalCaption.count()) > 0);

  const originalCanvas = originalCaption.locator('xpath=preceding-sibling::canvas[1]');
  check('Канвас исходного изображения найден рядом с подписью', (await originalCanvas.count()) > 0);

  await page.waitForTimeout(500); // дать время дорисоваться (тайлы читаются асинхронно)
  const stats = await originalCanvas.evaluate((canvas) => {
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    const data = ctx.getImageData(0, 0, width, height).data;
    let sum = 0;
    let sumSq = 0;
    let n = 0;
    const uniqueColors = new Set();
    for (let i = 0; i < data.length; i += 4) {
      const lum = (data[i] + data[i + 1] + data[i + 2]) / 3;
      sum += lum;
      sumSq += lum * lum;
      n++;
      if (n % 37 === 0) uniqueColors.add(`${data[i]},${data[i + 1]},${data[i + 2]}`);
    }
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    return { mean, stddev: Math.sqrt(Math.max(0, variance)), uniqueColors: uniqueColors.size, width, height };
  });
  console.log('Original thumbnail stats:', stats);
  check('Средняя яркость исходного изображения заметно выше чёрного фона', stats.mean > 40, `mean=${stats.mean.toFixed(1)}`);
  check('В исходном изображении есть разброс цветов (не однородная заливка)', stats.stddev > 5, `stddev=${stats.stddev.toFixed(1)}`);
  check('В исходном изображении несколько разных цветов (тайлы реально считаны)', stats.uniqueColors > 3, `uniqueColors=${stats.uniqueColors}`);

  // ---------- 3. Экспорт PDF не должен падать ----------
  let pdfError = false;
  page.once('pageerror', () => {
    pdfError = true;
  });
  const [download] = await Promise.all([
    page.waitForEvent('download', { timeout: 15000 }).catch(() => null),
    page.getByRole('button', { name: /Экспорт PDF/ }).click(),
  ]);
  check('Экспорт PDF завершился без ошибок страницы и с файлом загрузки', Boolean(download) && !pdfError);

  await browser.close();
  fs.unlinkSync(tmpPath);

  const failed = results.filter((r) => !r.ok);
  console.log('\n=== ИТОГО ===');
  console.log(`Пройдено: ${results.length - failed.length}/${results.length}`);
  if (failed.length) {
    console.log('Провалено:');
    for (const f of failed) console.log(` - ${f.name}${f.extra ? ' (' + f.extra + ')' : ''}`);
    process.exitCode = 1;
  }
}

main().catch((e) => {
  console.error('FATAL', e);
  process.exit(1);
});
