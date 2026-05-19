# Dual-View Ejection Fraction Prediction Model

A multi-task deep learning model for left ventricle segmentation and ejection fraction (EF prediction from pediatric echocardiogram videos. Trained on the EchoNet Pediatric dataset using a dual-view (A4C + PSAX) architecture with a two-phase training strategy.

## Test Metrics
| Metric | Score |
|--------|-------|
| Test MAE | 4.43 |
| Test RMSE | 6.2825 |
| R² | 0.6940 |
| Dice Score | 0.89 |
| AUC | 0.94 |

## Demo

<p>
  <img src="results/a4c_video.gif" width="300" alt="A4C Segmentation"/>
  <img src="results/psax_video.gif" width="300" alt="PSAX Segmentation"/>
</p>

## Results

![Training Curves](results/training-curves.png)
![Scatter Plot](results/Scatterplot.png)
![ROC Curve](results/ROC.png)