## 🧠 Shell-E: Smart Shelf Monitoring with Computer Vision & AI

Shell-E is an intelligent shelf analytics system designed to **automate planogram compliance** through **computer vision** and **AI-assisted detection**.

Developed for the **FEMSA Hackathon 2025**, the system enables retail teams to **detect missing, misplaced, or excess products** in shelf displays by simply uploading a photo and a CSV planogram file.

🏆 This project was awarded among top participants for its technical depth, real-world scalability, and retail impact.

---

### 🔍 Problem Statement

Traditional planogram auditing is manual, error-prone, and time-consuming. Retail teams need a faster, scalable way to detect:

- Which products are missing or misplaced
- If shelves are compliant with expected layouts
- How to prioritize restocking efficiently

---

### 💡 Our Solution

We developed **Shell-E**, a modular computer vision app that performs:

- **Real-time object detection** using the Roboflow API
- **Planogram parsing** from structured CSV files
- **Visual overlay generation** with PIL and OpenCV
- **Heatmaps, compliance KPIs, and shelf-space analytics**

All of this is delivered through a **lightweight Streamlit UI**, making it accessible to both technical and non-technical users.

---

### 🧠 Technologies Used

- `Streamlit` – front-end interface
- `Roboflow` – object detection model API
- `OpenCV` & `Pillow (PIL)` – image processing
- `pandas`, `numpy` – data wrangling
- `plotly`, `matplotlib` – visual analytics

---

### 🔬 Core Features

- 📷 Upload shelf image + CSV planogram  
- 🔎 Automatically identify each product's location and class  
- 🚦 Flag missing (red), misplaced (blue), and excess items (yellow)  
- 📊 View KPIs: compliance %, missing count, misplacements  
- 🔥 Generate product heatmaps and status overlays  
- 💬 Actionable suggestions for restocking or repositioning

---

### 📂 File Structure

- `shell-e.py` – Main app interface (Streamlit)
- `visualize_missing.py` – Visualization engine for missing products
- `Shell-E_Documentacion.pdf` – User and technical documentation


