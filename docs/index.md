# Chest X-Ray Image Classification

A deep learning application that classifies chest X-ray images into **COVID-19**, **Normal**, **Pneumonia**, and **Tuberculosis** using a MobileNetV3 transfer learning model (92% accuracy), with Grad-CAM heatmap explainability.

## Quick Links

- **[Live Demo](https://chest-xray-classification.azurewebsites.net/)** — upload an X-ray and get a classification with Grad-CAM overlay
- **[Architecture](architecture.md)** — package layout, data flow, and component design
- **[Developer Guide](dev-guide.md)** — environment setup, running the app, Docker, CI/CD
- **[GitHub Repository](https://github.com/JustaKris/CNN-Classification-of-Chest-X-Ray-Images)**

## Highlights

| | |
| --- | --- |
| **Model** | MobileNetV3Small fine-tuned on 7,135 chest X-rays |
| **Explainability** | Grad-CAM heatmaps show which image regions drive predictions |
| **Input validation** | CLIP zero-shot screening rejects non-X-ray images |
| **Stack** | TensorFlow/Keras, Flask, Docker, GitHub Actions, Azure App Service |

!!! note "Performance"
    The live demo runs on Azure's free tier. Expect **1–2 min cold start** after inactivity and **10–20 sec per prediction** (CLIP validation + classification + Grad-CAM generation).
