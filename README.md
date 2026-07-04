# Ore segmentation (DINO + GMM)

Script: `ml/intensity_gmm_segment.py`

## Web app (backend + frontend)

```bash
conda activate real
cd /home/human/bassel/real/real
ML_SERVICE_CONFIG=configs/ml_service.yaml python -m services.api.main
```

```bash
# separate terminal
cd /home/human/bassel/real/real/ore-classifier && npm run dev
```

Open the app:

- On this machine: http://localhost:5173
- From another device on the network: http://192.168.0.214:5173 (or http://santamaria:5173)

The dev server binds to `0.0.0.0:5173`; API calls are proxied to the backend on port 8000 on the same host. Ensure your firewall allows inbound TCP **5173** (and **8000** only if you call the API directly).

Leave **«Демо: локальная имитация ML»** off in the app bar to use the real backend.

### Public URL (anyone on the internet)

After backend + frontend are running, start a temporary Cloudflare tunnel (third terminal):

```bash
bash scripts/public-tunnel.sh
```

Share the printed `https://….trycloudflare.com` link from the terminal (copy the URL when the tunnel starts — old links stop working when the tunnel stops).

## Run

Single image (default test image):

```bash
python ml/intensity_gmm_segment.py
```

Single image (explicit path):

```bash
python ml/intensity_gmm_segment.py \
  --input  "Задача 3. Скажи мне, кто твой шлиф/Фото руд по сортам. ч1/Труднообогатимые руды/2539444-1.JPG" \
  --output-dir outputs/intensity_gmm_segment
```

Batch (all images under a folder):

```bash
python ml/intensity_gmm_segment.py \
  --source "task3-data/Фото руд по сортам. ч1" \
  --output-dir outputs/intensity_gmm_segment
```

With intermediate debug outputs (GMM histograms, masks):

```bash
python ml/intensity_gmm_segment.py --log-intermediates
```
