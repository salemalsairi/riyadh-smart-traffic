# 🚦 Riyadh Smart Traffic Control & Urban Intelligence System

An end-to-end Data Science and Urban Computing system that simulates adaptive traffic signal control (inspired by the **SCATS** methodology) and integrates a localized Retrieval-Augmented Generation (RAG) framework.

## 📖 Overview
This project aims to optimize urban mobility in Riyadh by addressing severe congestion dynamically. Instead of relying on static traffic light timers, it uses a convex delay optimization framework to reallocate green-light intervals based on historical traffic telemetry. 

Furthermore, the system features a built-in AI assistant that cross-references live metrics with local transportation policies and research.

## 🚀 Key Results & Performance
* **34.7% Delay Reduction:** Simulated adaptive control successfully smoothed traffic flow, reducing overall delays by 34.7% compared to naive static signal cycling.
* **The "Asymmetry Rule" Verified:** The algorithm demonstrated peak efficiency (28% - 31% flow optimization) during unbalanced directional loads, rather than uniform saturation.
* **Semantic AI Integration:** The local RAG engine successfully grounds traffic data in local policy (e.g., automatically identifying that severe congestion correlates to a 29% increase in passenger vehicle fuel consumption based on KAPSARC embedded literature).

## 🛠️ Tech Stack & Architecture
* **Data Processing & Simulation:** Python, Pandas, NumPy
* **Local Large Language Model:** `Llama 3` (8B) via Ollama
* **Embeddings & Vector Search:** `bge-m3` via Ollama, ChromaDB (Local)
* **Dashboard & Visualization:** Streamlit, Plotly

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/salemalsairi/riyadh-smart-traffic.git
   cd riyadh-smart-traffic
   
