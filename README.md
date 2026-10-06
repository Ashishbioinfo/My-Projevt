# Sentinel-2 Construction Change Detector

A Python baseline for finding **new construction between two Sentinel-2 dates**. It trains a Random Forest on labeled pixels and predicts a binary mask plus a per-pixel probability GeoTIFF. It detects spectral patterns consistent with construction; it cannot determine whether a building is legally permitted.

## Data needed

Prepare three co-registered GeoTIFFs in the same CRS, pixel grid, and dimensions:

- `before.tif`: six bands in this exact order: B02, B03, B04, B08, B11, B12.
- `after.tif`: the same six bands and order from a later date.
- `labels.tif`: one band, with `0` for not-new-construction, `1` for new construction, and NoData for unlabeled pixels.

Use atmospherically corrected surface reflectance from comparable seasons when possible. Apply cloud/shadow masking before training and mark excluded pixels as NoData in both imagery and labels. Resample bands to a common grid before stacking (B02/B03/B04/B08 are 10 m; B11/B12 are 20 m). The script requires aligned grids; it does not download, harmonize, or cloud-mask imagery.

Training reads the labeled pair into memory, so start with manageable training chips or an area of interest rather than a national-scale mosaic. The prediction step processes raster blocks.

## Install

Python 3.10 or later is recommended. In the VS Code terminal:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Rasterio reports an incompatible PROJ database on Windows, point the current terminal to Rasterio's bundled data before running the commands:

```powershell
$env:PROJ_LIB = (& python -c "import rasterio; print(rasterio.__path__[0])") + "\proj_data"
```

## Dashboard

Nagpur uses the supplied zone coordinates and hides the City selector. Choose a past T1 date and a more recent T2 date; the dashboard searches the public Sentinel-2 L2A catalog near each target date and shows the nearest available scene, including acquisition date and cloud cover. T1 and T2 are interactive Leaflet previews centered on the selected coordinates, each covering a 1 km radius (2 km across). Their metric-only scale bar is capped at 200 px and the map zoom is clamped to levels intended to show about 200 m to 1 km. The imagery comparison is a preview, not a trained construction prediction.

The comparison panel also provides exploratory L2A spectral-change layers. It uses the Scene Classification Layer to retain clear land pixels (classes 4 and 5), then shows:

- **Land-clearing candidates:** NDVI decrease of at least 0.25 between T1 and T2.
- **Built-surface gain candidates:** NDBI increase of at least 0.15 together with NDVI decrease of at least 0.10.
- **Combined candidates:** both layers, with built-surface candidates highlighted separately.

These fixed-threshold indicators are screening signals only. Crop cycles, bare soil, moisture, shadows, and residual cloud can create candidates. Sentinel-2 multispectral data cannot measure building height; the built-surface layer is not a vertical-building detector or a finding of illegality. Use the trained, locally validated model and permit records for further review.

T1/T2 and candidate layers are shown in separate Leaflet maps. The zone and anomaly table/PDF provide links that open the current Google satellite view separately. Google links do not provide historical Google imagery for the selected T1/T2 dates, and Google imagery is not embedded or copied into the PDF. A village-level model detection mask still requires configured village rasters and a trained model.

After loading T1 and T2, use **Download zone report PDF** below the monitoring result. Page one shows the T1/T2 overview images and only the total anomaly count. Following pages include scene details and the area summary, then each qualifying anomaly's coordinates, approximate area, surrounding-context T1/T2 crops with a red candidate outline, a current Google satellite link, and the municipal permit-verification notice. All qualifying patches are listed; isolated single-pixel noise is excluded. Google imagery is linked, not copied into the PDF. Spectral and model candidates are not legal determinations. Sentinel-2's 10 m sampling limits building-level detail; sharper inspection requires licensed high-resolution imagery.

## Train from Sentinel-2 L2A

The dashboard preview search is restricted to the Microsoft Planetary Computer `sentinel-2-l2a` collection; it has no Level-1C fallback. These dashboard previews are small RGB chips for viewing, not training inputs. The trainer consumes full-resolution, aligned six-band GeoTIFFs prepared from L2A bottom-of-atmosphere surface reflectance.

Prepare the training files as follows:

- Select before/after L2A scenes from comparable seasons and the same area. Keep reflectance scaling consistent between dates.
- Create each six-band raster in this exact order: B02, B03, B04, B08, B11, B12. Use a common CRS/grid/resolution; B11 and B12 are 20 m and need resampling to the chosen grid if using 10 m.
- Mask clouds, cirrus, cloud shadows, snow, and invalid pixels using the L2A Scene Classification Layer (SCL); mark excluded pixels NoData. Atmospheric correction does not remove clouds or shadows.
- In QGIS or another GIS, create a one-band label raster aligned exactly to the images: `0` = no new construction, `1` = new construction between the two dates, and NoData for unlabeled pixels. Label both classes across multiple spatial blocks and sites using reliable reference imagery/records.
- Do not use the RGB preview images or Google Maps screenshots as model inputs; the model requires the six spectral bands and labels.

Then train the Random Forest:

```powershell
python src/detect_construction.py train `
  --before data/before.tif `
  --after data/after.tif `
  --labels data/labels.tif `
  --model models/construction_rf.joblib
```

Evaluation uses spatial blocks for its holdout split to reduce the overly optimistic scores that can result from randomly splitting neighboring pixels. Label both classes across multiple 128-pixel blocks and, ideally, across multiple sites. The reported score is a development check, not proof of accuracy in a different region or season.

## Predict

```powershell
python src/detect_construction.py predict `
  --before data/before_new.tif `
  --after data/after_new.tif `
  --model models/construction_rf.joblib `
  --output outputs/new_construction.tif `
  --threshold 0.5
```

The output mask uses `1` for detected new construction, `0` for not detected, and `255` for NoData. A companion `*_probability.tif` contains the model probability for class 1. Adjust the threshold against independently labeled validation sites; the default is not universally calibrated.

## Important limitations

- The model needs representative, manually labeled examples. No trained weights or satellite imagery are bundled.
- Construction is inferred from temporal spectral change, so bare soil, demolition, seasonal effects, shadows, and registration errors can cause false detections.
- Use independent, geographically separated validation data before operational use. Review detections with higher-resolution imagery and local permitting records before taking action.
- Keep imagery reflectance scaling consistent across training and prediction. Random Forests do not require a particular numeric scale, but inconsistent products can change learned thresholds.
