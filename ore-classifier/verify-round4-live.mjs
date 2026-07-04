// Живая проверка правок "раунд 4" в реальном браузере через Playwright.
// Запуск: node verify-round4-live.mjs (dev-сервер должен быть поднят на localhost:5173)
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

  // ---------- 0. Сброс демо-данных, чтобы гарантированно получить чистый детерминированный сид ----------
  await page.goto(BASE_URL, { waitUntil: 'networkidle' });
  await page.locator('button:has(svg[data-testid="MoreVertIcon"])').click();
  await page.getByText('Сбросить демо-данные').click();
  await page.getByRole('button', { name: 'Сбросить' }).click();
  await page.waitForTimeout(5000);
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);

  // ---------- 1. Дашборд: читаемая горизонтальная диаграмма со всеми 12 месторождениями ----------
  await page.goto(`${BASE_URL}/dashboard`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  const barChartSection = page
    .locator('text=Классы руды по месторождениям')
    .locator('xpath=ancestor::*[contains(@class,"MuiPaper-root")][1]');
  const yAxisLabels = await barChartSection.locator('.MuiChartsAxis-tickLabel').allTextContents().catch(() => []);
  const expectedNames = [
    'Октябрьское', 'Талнахское', 'Норильск-1', 'Ждановское', 'Заполярное', 'Тундровое',
    'Спутник', 'Верхнее', 'Котсельваара-Каммикиви', 'Семилетка', 'Быстринское', 'Nkomati',
  ];
  const visibleNames = expectedNames.filter((n) => yAxisLabels.some((l) => l.includes(n)));
  check(
    'Все 12 названий месторождений видны на диаграмме (не обрезаны)',
    visibleNames.length === 12,
    `видно: ${visibleNames.length}/12 — ${JSON.stringify(yAxisLabels)}`,
  );
  const chartBox = await barChartSection.boundingBox().catch(() => null);
  check('Высота диаграммы увеличена под 12 месторождений', Boolean(chartBox && chartBox.height >= 300), `height=${chartBox?.height}`);

  // ---------- 2. Редактор кадра: превью контура полигона/лассо + живой пересчёт метрик ----------
  await page.goto(`${BASE_URL}/`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  const firstRow = page.locator('.MuiDataGrid-row').first();
  let experimentHref = null;
  if (await firstRow.count()) {
    const rowId = await firstRow.getAttribute('data-id');
    experimentHref = `/experiments/${rowId}`;
    await firstRow.locator('.MuiDataGrid-cell[data-field="title"] span').click();
    await page.waitForTimeout(1200);
  }
  const frameThumbCanvas = page.locator('canvas').first();
  if (await frameThumbCanvas.count()) {
    await frameThumbCanvas.click();
  }
  await page.waitForTimeout(2500);
  check('Открылся редактор кадра', page.url().includes('/frames/'));

  const paintingOverlay = page.locator('[data-testid="painting-overlay"]');
  const previewCanvas = page.locator('canvas[data-testid="mask-preview-canvas"]');
  await page.waitForSelector('canvas[data-testid="mask-preview-canvas"]', { timeout: 8000 }).catch(() => {});
  check('Канвас превью контура присутствует в DOM', (await previewCanvas.count()) > 0);

  async function previewHasContent() {
    return previewCanvas.first().evaluate((canvas) => {
      const ctx = canvas.getContext('2d');
      const { width, height } = canvas;
      if (!width || !height) return false;
      const data = ctx.getImageData(0, 0, width, height).data;
      for (let i = 3; i < data.length; i += 4) {
        if (data[i] > 0) return true;
      }
      return false;
    });
  }

  // 2a. Полигон: несколько кликов должны рисовать резиновый контур на preview-канвасе
  await page.locator('button[value="polygon"]').click().catch(() => {});
  await page.waitForTimeout(150);
  if (await paintingOverlay.count()) {
    const box = await paintingOverlay.boundingBox();
    if (box) {
      const cx = box.x + box.width / 2;
      const cy = box.y + box.height / 2;
      await page.mouse.click(cx - 60, cy - 40);
      await page.waitForTimeout(80);
      await page.mouse.click(cx + 60, cy - 40);
      await page.waitForTimeout(80);
      await page.mouse.move(cx, cy + 60);
      await page.waitForTimeout(120);
    }
  }
  const polygonPreviewVisible = await previewHasContent().catch(() => false);
  check('Контур полигона виден на превью-канвасе во время рисования', polygonPreviewVisible);
  await page.keyboard.press('Escape').catch(() => {});
  await page.waitForTimeout(150);
  const polygonPreviewClearedAfterEscape = !(await previewHasContent().catch(() => true));
  check('Превью контура скрывается после Escape', polygonPreviewClearedAfterEscape);

  // 2b. Лассо: контур виден во время обводки от руки
  await page.locator('button[value="lasso"]').click().catch(() => {});
  await page.waitForTimeout(150);
  if (await paintingOverlay.count()) {
    const box = await paintingOverlay.boundingBox();
    if (box) {
      const cx = box.x + box.width / 2;
      const cy = box.y + box.height / 2;
      const r = Math.min(box.width, box.height) * 0.15;
      await page.mouse.move(cx + r, cy);
      await page.mouse.down();
      const steps = 16;
      for (let i = 1; i <= steps; i++) {
        const angle = (i / steps) * Math.PI * 1.5;
        await page.mouse.move(cx + r * Math.cos(angle), cy + r * Math.sin(angle), { steps: 2 });
      }
      const lassoPreviewVisible = await previewHasContent().catch(() => false);
      check('Контур лассо виден на превью-канвасе во время обводки', lassoPreviewVisible);
      await page.mouse.up();
      await page.waitForTimeout(200);
    }
  }
  const lassoPreviewClearedAfterRelease = !(await previewHasContent().catch(() => true));
  check('Превью контура скрывается после завершения лассо', lassoPreviewClearedAfterRelease);

  // 2c. Живой пересчёт метрик: проценты в легенде меняются во время рисования, до "Сохранить"
  // Сначала сохраняем текущие (тестовые) правки полигона/лассо, чтобы получить чистую базовую точку.
  const saveButton = page.getByRole('button', { name: /Сохранить изменения/ });
  if (!(await saveButton.isDisabled().catch(() => true))) {
    await saveButton.click();
    await page.waitForTimeout(500);
  }
  const saveDisabledBefore = await saveButton.isDisabled().catch(() => true);
  check('Кнопка "Сохранить изменения" недоступна сразу после сохранения', saveDisabledBefore);

  await page.locator('button[value="brush"]').click().catch(() => {});
  const legendPanel = page.locator('text=Слои маски и цвет кисти').locator('xpath=ancestor::*[contains(@class,"MuiPaper-root")][1]');
  // Выбираем класс "Тальк" (3-й радио в легенде: обычные/тонкие/тальк/матрица) — контрастный класс, изменение хорошо заметно.
  await legendPanel.locator('input[type="radio"]').nth(2).click({ force: true }).catch(() => {});
  for (let i = 0; i < 40; i++) await page.keyboard.press(']'); // максимальный радиус кисти
  await page.waitForTimeout(150);

  const legendTextBefore = await legendPanel.textContent().catch(() => '');

  if (await paintingOverlay.count()) {
    const box = await paintingOverlay.boundingBox();
    if (box) {
      // Несколько широких проходов кистью максимального радиуса по всему кадру — гарантированно заметное изменение площади.
      for (let row = 0; row < 4; row++) {
        const y = box.y + box.height * (0.15 + row * 0.25);
        await page.mouse.move(box.x + box.width * 0.1, y);
        await page.mouse.down();
        await page.mouse.move(box.x + box.width * 0.9, y, { steps: 15 });
        await page.mouse.up();
      }
      await page.waitForTimeout(300);
    }
  }
  const legendTextAfter = await legendPanel.textContent().catch(() => '');
  const saveDisabledAfter = await saveButton.isDisabled().catch(() => true);
  check('Кнопка "Сохранить изменения" стала доступна после правки (dirty)', !saveDisabledAfter);
  check(
    'Проценты в легенде изменились сразу после рисования кистью, без сохранения',
    legendTextBefore !== legendTextAfter,
    `до: ${legendTextBefore?.slice(0, 160)} | после: ${legendTextAfter?.slice(0, 160)}`,
  );
  const livePreviewCaption = await page.locator('text=Предварительно, по текущей').first().isVisible().catch(() => false);
  check('Панель класса кадра показывает пометку "предварительно" при несохранённых правках', livePreviewCaption);

  // ---------- 3. Отчёт: убраны "Запасы руды"/"Содержание металлов", есть компактная строка ----------
  if (experimentHref) {
    await page.goto(`${BASE_URL}${experimentHref}`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);
    await page.getByRole('button', { name: 'Сформировать отчёт' }).click();
    await page.waitForTimeout(1500);

    const reservesVisible = await page.locator('text=Запасы руды').first().isVisible().catch(() => false);
    check('В отчёте нет блока "Запасы руды"', !reservesVisible);
    const gradesVisible = await page.locator('text=Содержание металлов').first().isVisible().catch(() => false);
    check('В отчёте нет блока "Содержание металлов"', !gradesVisible);
    const depositInfoBlockVisible = await page.locator('text=Сведения о месторождении').first().isVisible().catch(() => false);
    check('В отчёте нет отдельного блока "Сведения о месторождении"', !depositInfoBlockVisible);

    const pageText = await page.textContent('body').catch(() => '');
    check('В отчёте есть компактная строка с рудным узлом', /рудный узел/i.test(pageText ?? ''));
    const mineralProfileVisible = await page.locator('text=Профиль минералов месторождения').first().isVisible().catch(() => false);
    check('В отчёте сохранён раздел "Профиль минералов месторождения"', mineralProfileVisible);
  } else {
    check('Проверка отчёта пропущена — не найден href эксперимента', false);
  }

  await browser.close();

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
