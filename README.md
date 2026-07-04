# Ore segmentation (DINO + GMM)

Script: `ml/intensity_gmm_segment.py`

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
