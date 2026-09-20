# Deep-Learning---Quiz-02
Receipt Detection Roll No: 2023-SE-08   Session:2023-27
# Receipt Detection Quiz

## Objective
Detect and separate multiple receipts from a single photo using a
pre-trained zero-shot object detection model (Grounding DINO), then
crop and save each receipt as its own image file.

## Approach
1. Upload an image containing multiple receipts (5 in this case).
2. Run Grounding DINO with text prompts ("a receipt", "a paper receipt",
   "a shopping receipt") to detect candidate regions.
3. Filter out detections with implausible size or aspect ratio to
   remove false positives.
4. Remove duplicate/overlapping boxes using IoU (Intersection over
   Union), keeping the highest-confidence detection.
5. If fewer than the expected number of receipts are found, retry
   detection at progressively lower confidence thresholds.
6. Crop each detected receipt from the original image and save it
   as a separate file.
7. Log each detection's confidence score and bounding box coordinates.
8. Package all cropped receipts and the log into a ZIP file.

## Model used
- **Grounding DINO (tiny)** — `IDEA-Research/grounding-dino-tiny`
  via Hugging Face Transformers
- Runs on CPU (no GPU required)

## Files in this folder
| File | Description |
|---|---|
| `notebook.ipynb` | Full Colab notebook, runnable top to bottom |
| `receipt_1.jpg` – `receipt_5.jpg` | Cropped individual receipts |
| `detection_log.txt` | Confidence score and box coordinates per receipt |

## How to run
1. Open `notebook.ipynb` in Google Colab.
2. Run all cells in order (Runtime → Run all).
3. Note: after the PyTorch reinstall cell, Colab requires a runtime
   restart — re-run the model-loading cell afterward before running
   the detection cell.
4. Upload the input image when prompted.

## Result
Successfully detected and separated all 5 receipts from the input
image with confidence scores logged in `detection_log.txt`.

## Author
Hajara Waseem
