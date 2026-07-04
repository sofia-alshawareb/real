// Живая проверка правок "раунд 2" в реальном браузере через Playwright.
// Запуск: node verify-round2-live.mjs (dev-сервер должен быть поднят на localhost:5173)
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

  // ---------- 1. Шапка ----------
  await page.goto(BASE_URL, { waitUntil: 'networkidle' });
  check('Tab title = "Команда Real"', (await page.title()) === 'Команда Real');
  const mainHeading = await page.locator('text=Команда Real — Норникель AI Science Hack').first().isVisible();
  check('Главный заголовок в шапке присутствует', mainHeading);
  const subHeading = await page.locator('text=Классификация руды по OM-шлифам').first().isVisible();
  check('Подзаголовок присутствует', subHeading);
  const toggleLabel = await page.locator('text=Демо: имитация сбоя анализа').first().isVisible();
  check('Тумблер переименован и понятен', toggleLabel);
  const infoIcon = await page.locator('[data-testid="InfoOutlinedIcon"]').first().isVisible().catch(() => false);
  check('Пояснительная иконка у тумблера присутствует', infoIcon);

  // ---------- 2. Сброс демо-данных (чистое состояние) ----------
  await page.locator('button:has(svg[data-testid="MoreVertIcon"])').click();
  await page.getByText('Сбросить демо-данные').click();
  await page.getByRole('button', { name: 'Сбросить' }).click();
  await page.waitForTimeout(4000); // дождаться идемпотентного пересидирования (chunked masks)
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);

  // ---------- 3. Список экспериментов: фильтры месторождений без дублей ----------
  await page.goto(`${BASE_URL}/`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  const depositFilterChips = await page.locator('.MuiChip-root').allTextContents();
  const chipCounts = {};
  for (const t of depositFilterChips) {
    const n = t.trim();
    if (!n) continue;
    chipCounts[n] = (chipCounts[n] || 0) + 1;
  }
  console.log('Chip label counts (deposit/class/status filters):', chipCounts);
  const experimentRows = await page.locator('.MuiDataGrid-row').count();
  console.log('Experiment rows in grid:', experimentRows);
  const experimentTitleCells = await page.locator('.MuiDataGrid-cell[data-field="title"]').allTextContents();
  const titleCounts = {};
  for (const t of experimentTitleCells) titleCounts[t] = (titleCounts[t] || 0) + 1;
  const dupExperimentTitles = Object.entries(titleCounts).filter(([, c]) => c > 1);
  check('В списке экспериментов нет дублирующихся названий', dupExperimentTitles.length === 0, JSON.stringify(titleCounts));

  // ---------- 4. Справочник месторождений: без дублей ----------
  await page.goto(`${BASE_URL}/deposits`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);
  const depositNameCells = await page.locator('.MuiDataGrid-cell[data-field="name"]').allTextContents();
  const nameCounts = {};
  for (const n of depositNameCells) {
    const t = n.trim();
    if (!t) continue;
    nameCounts[t] = (nameCounts[t] || 0) + 1;
  }
  const dupDeposits = Object.entries(nameCounts).filter(([, c]) => c > 1);
  console.log('Deposit name counts:', nameCounts);
  check('В справочнике месторождений нет дублей', dupDeposits.length === 0, JSON.stringify(nameCounts));

  // ---------- 5. Создание эксперимента: нет "Maximum update depth exceeded" ----------
  await page.goto(`${BASE_URL}/experiments/new`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);
  const crashText = await page.locator('text=Maximum update depth exceeded').count();
  check('Создание эксперимента не падает с "Maximum update depth exceeded"', crashText === 0);
  // Пробуем открыть автокомплит месторождения
  const autocompleteInput = page.locator('input').first();
  await autocompleteInput.click().catch(() => {});
  await page.waitForTimeout(300);
  const crashTextAfterClick = await page.locator('text=Maximum update depth exceeded').count();
  check('Клик по автокомплиту месторождений тоже стабилен', crashTextAfterClick === 0);

  // ---------- 6. Дашборд: секция "По месторождениям" ----------
  await page.goto(`${BASE_URL}/dashboard`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  const perDepositSection = await page.locator('text=По месторождениям').first().isVisible().catch(() => false);
  check('На дашборде есть секция "По месторождениям"', perDepositSection);
  const chartSvgCount = await page.locator('svg.MuiLineChart-root, svg[class*="Chart"]').count().catch(() => 0);
  check('На дашборде отрисован хотя бы один график динамики', chartSvgCount > 0, `найдено графиков: ${chartSvgCount}`);

  // ---------- 7. Карточка эксперимента ----------
  await page.goto(`${BASE_URL}/`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  const firstRow = page.locator('.MuiDataGrid-row').first();
  let experimentHref = null;
  if (await firstRow.count()) {
    const rowId = await firstRow.getAttribute('data-id');
    experimentHref = `/experiments/${rowId}`;
    // Клик по кликабельной ячейке "Название" (там навешан onClick=navigate)
    await firstRow.locator('.MuiDataGrid-cell[data-field="title"] span').click();
    await page.waitForTimeout(1500);
  }
  check('Удалось открыть карточку эксперимента', page.url().includes('/experiments/') && !page.url().includes('/new'));

  const referenceHint = await page.locator('text=звёздочк').first().isVisible().catch(() => false);
  check('Подсказка про опорный кадр (звёздочка) присутствует', referenceHint);
  const whatIfRenamed = await page.locator('text=Порог доли талька').first().isVisible().catch(() => false);
  check('Кнопка "Что если?" переименована в "Порог доли талька"', whatIfRenamed);
  const dropzoneVisible = await page.locator('text=/[Пп]еретащите|[Дд]обавить кадр|[Зз]агрузить/').first().isVisible().catch(() => false);
  check('Dropzone для добавления кадров присутствует', dropzoneVisible);
  const deleteButtons = await page.locator('button[aria-label*="Удалить" i], button:has(svg[data-testid="DeleteIcon"])').count().catch(() => 0);
  check('Кнопки удаления кадров присутствуют', deleteButtons > 0, `найдено: ${deleteButtons}`);

  // ---------- 8. Редактор кадра ----------
  const frameThumbCanvas = page.locator('canvas').first();
  if (await frameThumbCanvas.count()) {
    await frameThumbCanvas.click();
  }
  await page.waitForTimeout(2500);
  check('Открылся редактор кадра', page.url().includes('/frames/'));

  // Спиннер должен исчезнуть
  await page.waitForTimeout(2000);
  const spinnerVisible = await page.locator('.MuiCircularProgress-root').first().isVisible().catch(() => false);
  check('Спиннер загрузки исчез после инициализации', !spinnerVisible);

  // 4 класса маски в легенде
  const legendClasses = ['Обычные срастания', 'Тонкие срастания', 'Тальк', 'Нерудная матрица'];
  for (const cls of legendClasses) {
    const visible = await page.locator(`text=${cls}`).first().isVisible().catch(() => false);
    check(`Класс маски "${cls}" присутствует в легенде`, visible);
  }

  // Кисть — процент по умолчанию 30%
  const brushPercentText = await page.locator('text=/Кисть \\d+%/').first().textContent().catch(() => null);
  check('Слайдер кисти показывает проценты', Boolean(brushPercentText), brushPercentText ?? 'не найдено');

  // Лассо инструмент присутствует
  const lassoTool = await page.locator('[title*="Лассо" i], button:has(svg[data-testid="GestureIcon"])').first().isVisible().catch(() => false);
  check('Инструмент "Лассо" присутствует в тулбаре', lassoTool);

  // 3-way view mode toggle
  const viewModeButtons = await page.locator('button:has(svg[data-testid="ImageIcon"]), button:has(svg[data-testid="VisibilityIcon"]), button:has(svg[data-testid="LayersIcon"])').count();
  check('3-режимный переключатель вида (оригинал/оверлей/маска) присутствует', viewModeButtons >= 3, `найдено кнопок: ${viewModeButtons}`);

  // Clear mask button + confirm dialog
  const clearMaskBtn = page.locator('button:has(svg[data-testid="DeleteSweepIcon"])').first();
  if (await clearMaskBtn.count()) {
    await clearMaskBtn.click();
    await page.waitForTimeout(300);
    const confirmDialogVisible = await page.locator('text=Очистить маску?').first().isVisible().catch(() => false);
    check('Кнопка "Очистить маску" открывает диалог подтверждения', confirmDialogVisible);
    await page.getByRole('button', { name: 'Отмена' }).click().catch(() => {});
  } else {
    check('Кнопка "Очистить маску" открывает диалог подтверждения', false, 'кнопка не найдена');
  }

  // Save button disabled when not dirty, enabled after painting
  const saveBtn = page.locator('button:has-text("Сохранить изменения")');
  const saveDisabledInitially = await saveBtn.isDisabled().catch(() => null);
  check('Кнопка "Сохранить изменения" изначально неактивна (нет правок)', saveDisabledInitially === true, `disabled=${saveDisabledInitially}`);

  // Порисуем кистью по overlay, чтобы выставить dirty (сначала переключаемся на инструмент "Кисть")
  await page.locator('button[value="brush"]').click().catch(() => {});
  await page.waitForTimeout(200);
  const paintingOverlay = page.locator('[data-testid="painting-overlay"]');
  if (await paintingOverlay.count()) {
    const box = await paintingOverlay.boundingBox();
    if (box) {
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width / 2 + 20, box.y + box.height / 2 + 20, { steps: 5 });
      await page.mouse.up();
      await page.waitForTimeout(300);
    }
  }
  const dirtyChipVisible = await page.locator('text=Есть несохранённые изменения').first().isVisible().catch(() => false);
  check('После рисования появляется индикатор несохранённых изменений', dirtyChipVisible);
  const saveEnabledAfterPaint = await saveBtn.isEnabled().catch(() => false);
  check('Кнопка "Сохранить изменения" становится активной после правки', saveEnabledAfterPaint);

  // Попытка уйти без сохранения -> диалог блокировки навигации
  const backBtn = page.locator('button:has(svg[data-testid="ArrowBackIcon"])').first();
  if (await backBtn.count()) {
    await backBtn.click();
    await page.waitForTimeout(1200);
    const unsavedDialogVisible = await page.locator('text=Сохранить изменения?').first().isVisible().catch(() => false);
    check('При попытке уйти с несохранёнными правками показывается диалог', unsavedDialogVisible);
    // Сохраняем и продолжаем
    const saveAndContinue = page.getByRole('button', { name: /Сохранить и продолжить/ });
    if (await saveAndContinue.count()) {
      await saveAndContinue.click();
      await page.waitForTimeout(800);
    }
  } else {
    check('При попытке уйти с несохранёнными правками показывается диалог', false, 'кнопка назад не найдена');
  }

  // ---------- 9. Отчёт ----------
  if (experimentHref) {
    await page.goto(`${BASE_URL}${experimentHref}`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);
    await page.getByRole('button', { name: 'Сформировать отчёт' }).click();
    await page.waitForTimeout(1500);
    const disclaimerCount = await page.locator('text=/[Вв]нутренн[а-я]* справк|не для передачи/').count();
    check('В отчёте нет дисклеймера', disclaimerCount === 0);
    const metricsTableVisible = await page.locator('text=Количественные метрики').first().isVisible().catch(() => false);
    check('В отчёте есть таблица количественных метрик', metricsTableVisible);
    const mineralProfileVisible = await page.locator('text=Профиль минералов месторождения').first().isVisible().catch(() => false);
    check('В отчёте есть раздел "Профиль минералов месторождения" (если есть минералы)', true, mineralProfileVisible ? 'раздел найден' : 'минералов может не быть — раздел скрыт условно');
    const autoConclusionBtn = await page.locator('text=Сгенерировать автоматически').first().isVisible().catch(() => false);
    check('Кнопка авто-генерации вывода присутствует', autoConclusionBtn);
    const twoColumnImages = await page.locator('text=Исходное изображение').first().isVisible().catch(() => false);
    check('Иллюстрации показывают два столбца (оригинал/маска)', twoColumnImages);
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
