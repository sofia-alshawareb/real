# Классификация руды по OM-шлифам

Веб-приложение для загрузки панорам шлифов, автоматической сегментации и ручной проверки разметки.

## Установка

### Backend (Python)

```bash
cd /path/to/real
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Если используете Conda, активируйте своё окружение вместо создания `.venv`.

Модели DINOv2 должны лежать в `data/models/` (репозиторий `dinov2` и файл весов `checkpoints/dinov2_vits14_reg4_pretrain.pth`).

### Frontend (Node.js)

```bash
cd ore-classifier
npm install
```

## Запуск

Нужны **два терминала**.

**1. Backend** — из корня проекта:

```bash
ML_SERVICE_CONFIG=configs/ml_service.yaml python -m services.api.main
```

**2. Frontend**:

```bash
cd ore-classifier
npm run dev
```

Откройте в браузере: **http://localhost:5173**

Чтобы работала серверная ML-сегментация, в верхней панели приложения **выключите** переключатель «Демо: локальная имитация ML».

Подробное описание интерфейса — в [ore-classifier/docs/РУКОВОДСТВО_ПОЛЬЗОВАТЕЛЯ.md](ore-classifier/docs/РУКОВОДСТВО_ПОЛЬЗОВАТЕЛЯ.md).
