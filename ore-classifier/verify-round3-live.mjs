// Живая проверка правок "раунд 3" в реальном браузере через Playwright.
// Запуск: node verify-round3-live.mjs (dev-сервер должен быть поднят на localhost:5173)
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

  // ---------- 0. Сброс демо-данных, чтобы гарантированно получить новый сид v2 ----------
  await page.goto(BASE_URL, { waitUntil: 'networkidle' });
  await page.locator('button:has(svg[data-testid="MoreVertIcon"])').click();
  await page.getByText('Сбросить демо-данные').click();
  await page.getByRole('button', { name: 'Сбросить' }).click();
  await page.waitForTimeout(5000); // структурный сид + быстрые предварительные метрики
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);

  // ---------- 1. Справочник месторождений: 12 месторождений из PDF, без дублей, новые поля ----------
  await page.goto(`${BASE_URL}/deposits`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);
  const depositNameCells = await page.locator('.MuiDataGrid-cell[data-field="name"]').allTextContents();
  const names = depositNameCells.map((n) => n.trim()).filter(Boolean);
  const nameCounts = {};
  for (const n of names) nameCounts[n] = (nameCounts[n] || 0) + 1;
  const dupDeposits = Object.entries(nameCounts).filter(([, c]) => c > 1);
  console.log('Deposit names:', names);
  check('В справочнике ровно 12 месторождений', names.length === 12, `найдено: ${names.length}`);
  check('В справочнике месторождений нет дублей', dupDeposits.length === 0, JSON.stringify(nameCounts));
  const expectedNames = [
    'Октябрьское', 'Талнахское', 'Норильск-1', 'Ждановское', 'Заполярное', 'Тундровое',
    'Спутник', 'Верхнее', 'Котсельваара-Каммикиви', 'Семилетка', 'Быстринское', 'Nkomati',
  ];
  const missing = expectedNames.filter((n) => !names.includes(n));
  check('Все 12 месторождений из PDF присутствуют', missing.length === 0, `отсутствуют: ${JSON.stringify(missing)}`);
  const oreClusterColVisible = await page.locator('text=Рудный узел').first().isVisible().catch(() => false);
  check('В таблице справочника есть колонка "Рудный узел"', oreClusterColVisible);
  const balanceColVisible = await page.locator('text=Балансовые запасы').first().isVisible().catch(() => false);
  check('В таблице справочника есть колонка "Балансовые запасы"', balanceColVisible);

  // Открыть карточку редактирования месторождения — проверить новые поля формы
  const editButtons = page.locator('button:has(svg[data-testid="EditIcon"])');
  if (await editButtons.count()) {
    await editButtons.first().click();
    await page.waitForTimeout(400);
    const clusterField = await page.locator('label:has-text("Рудный узел")').first().isVisible().catch(() => false);
    check('Диалог редактирования месторождения содержит поле "Рудный узел"', clusterField);
    const reservesField = await page.locator('label:has-text("Балансовые запасы")').first().isVisible().catch(() => false);
    check('Диалог редактирования месторождения содержит поле запасов', reservesField);
    const gradesField = await page.locator('label:has-text("Никель")').first().isVisible().catch(() => false);
    check('Диалог редактирования месторождения содержит поле содержания металлов', gradesField);
    await page.getByRole('button', { name: 'Отмена' }).click().catch(() => {});
  } else {
    check('Диалог редактирования месторождения содержит поле "Рудный узел"', false, 'кнопка редактирования не найдена');
  }

  // ---------- 2. Дашборд: графики по месторождениям непустые сразу после сида ----------
  await page.goto(`${BASE_URL}/dashboard`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  const perDepositSection = await page.locator('text=По месторождениям').first().isVisible().catch(() => false);
  check('На дашборде есть секция "По месторождениям"', perDepositSection);
  const emptyStateVisible = await page.locator('text=Пока недостаточно данных по кадрам').first().isVisible().catch(() => false);
  check('Секция "По месторождениям" не пуста сразу после сида', !emptyStateVisible);
  const chartCards = await page.locator('text=динамика экспериментов и средняя доля талька').count().catch(() => 0);
  check('Есть хотя бы 2 карточки динамики по месторождениям сразу после сида', chartCards >= 2, `найдено: ${chartCards}`);
  // Проверяем, что в тексте графика видна ненулевая доля талька хотя бы у одного месторождения
  const pageContent = await page.content();
  const hasNonZeroPercent = /\d+[.,]\d%/.test(pageContent) || /[1-9]\d?%/.test(pageContent);
  check('На дашборде отображаются ненулевые проценты (метрики посчитаны сразу)', hasNonZeroPercent);

  // ---------- 3. Редактор шлифа: заливка + лассо ----------
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
  await page.waitForSelector('button[value="fill"]', { timeout: 5000 }).catch(() => {});

  const fillTool = page.locator('button[value="fill"]');
  check('Инструмент "Заливка" присутствует в тулбаре', await fillTool.count() > 0);
  const lassoTool = page.locator('button[value="lasso"]');
  check('Инструмент "Лассо" присутствует в тулбаре', await lassoTool.count() > 0);

  // Заливка помечает маску dirty
  await fillTool.click().catch(() => {});
  await page.waitForTimeout(200);
  const paintingOverlay = page.locator('[data-testid="painting-overlay"]');
  if (await paintingOverlay.count()) {
    const box = await paintingOverlay.boundingBox();
    if (box) {
      await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
      await page.waitForTimeout(300);
    }
  }
  let dirtyChipVisible = await page.locator('text=Есть несохранённые изменения').first().isVisible().catch(() => false);
  check('Заливка помечает маску как изменённую (dirty)', dirtyChipVisible);

  // Отменяем правку заливки, чтобы протестировать лассо с чистого состояния
  await page.keyboard.press('Control+z').catch(() => {});
  await page.waitForTimeout(300);

  // Лассо: обводим область от руки (drag) и проверяем, что маска тоже помечается dirty
  await lassoTool.click().catch(() => {});
  await page.waitForTimeout(200);
  if (await paintingOverlay.count()) {
    const box = await paintingOverlay.boundingBox();
    if (box) {
      const cx = box.x + box.width / 2;
      const cy = box.y + box.height / 2;
      const r = Math.min(box.width, box.height) * 0.15;
      await page.mouse.move(cx + r, cy);
      await page.mouse.down();
      const steps = 24;
      for (let i = 1; i <= steps; i++) {
        const angle = (i / steps) * Math.PI * 2;
        await page.mouse.move(cx + r * Math.cos(angle), cy + r * Math.sin(angle), { steps: 2 });
      }
      await page.mouse.up();
      await page.waitForTimeout(300);
    }
  }
  dirtyChipVisible = await page.locator('text=Есть несохранённые изменения').first().isVisible().catch(() => false);
  check('Лассо (обводка от руки) помечает маску как изменённую (dirty)', dirtyChipVisible);

  // ---------- 4. Отчёт: редактируемые заметки по минералам ----------
  if (experimentHref) {
    await page.goto(`${BASE_URL}${experimentHref}`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);
    await page.getByRole('button', { name: 'Сформировать отчёт' }).click();
    await page.waitForTimeout(1500);
    // Раунд 4: отдельный блок "Сведения о месторождении" убран, вместо него — компактная строка в шапке
    const depositInfoBlockVisible = await page.locator('text=Сведения о месторождении').first().isVisible().catch(() => false);
    check('В отчёте нет отдельного блока "Сведения о месторождении"', !depositInfoBlockVisible);
    const pageText = await page.textContent('body').catch(() => '');
    check('В отчёте отображается рудный узел (компактная строка)', /рудный узел/i.test(pageText ?? ''));

    const mineralNoteField = page
      .locator('tr')
      .filter({ has: page.locator('text=Сульфид') })
      .locator('input')
      .first();
    if (await mineralNoteField.count()) {
      await mineralNoteField.fill('Тестовая заметка для проверки');
      await page.waitForTimeout(1200);
      const value = await mineralNoteField.inputValue();
      check('Заметка по минералу редактируется в отчёте', value === 'Тестовая заметка для проверки', value);
      await page.reload({ waitUntil: 'networkidle' });
      await page.waitForTimeout(1500);
      const mineralNoteFieldAfterReload = page
        .locator('tr')
        .filter({ has: page.locator('text=Сульфид') })
        .locator('input')
        .first();
      const persistedValue = await mineralNoteFieldAfterReload.inputValue().catch(() => '');
      check('Отредактированная заметка сохраняется в черновике отчёта после перезагрузки', persistedValue === 'Тестовая заметка для проверки', persistedValue);
    } else {
      check('Заметка по минералу редактируется в отчёте', false, 'поле заметки не найдено (возможно нет минералов с ролью "Сульфид")');
    }
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
