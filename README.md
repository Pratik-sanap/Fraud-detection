# Fraud Detection Project

A full-stack fraud analytics dashboard built with:

- Frontend: React + Vite + Recharts
- Backend: Flask + ML model inference
- Data/ML notebooks and training scripts

The app lets you upload a CSV, run fraud detection, and view quality metrics, fraud summary, charts, and flagged transaction insights.

## Project Structure

- backend: Flask API and model loading
- frontend: React dashboard UI
- data: sample and processed CSV files
- ml: notebooks and model training scripts

## Prerequisites

- Python 3.10+ (tested with 3.11)
- Node.js 18+ and npm

## Quick Start

### 1) Install frontend dependencies

From project root:

```powershell
npm install --prefix frontend
```

### 2) Install backend dependencies

From project root:

```powershell
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

If your virtual environment does not exist yet:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

### 3) Run backend

```powershell
Set-Location backend
..\.venv\Scripts\python.exe app.py
```

Backend starts on:

- http://127.0.0.1:5000

### 4) Run frontend

In a separate terminal, from project root:

```powershell
npm run dev
```

Frontend starts on:

- http://localhost:5173

## How to Use

1. Open the frontend URL.
2. Choose a CSV file in the upload area.
3. Click Upload & Detect Fraud.
4. Review:
   - Data quality report
   - Fraud summary
   - Fraud charts
   - Flagged transaction table

## API Endpoints

### GET /

Simple API status string.

### GET /health

Returns backend health and model metadata.

### POST /upload

Accepts multipart form-data with field name file.

Response example:

```json
{
  "total_transactions": 1447,
  "fraud_detected": 21
}
```

## Root Scripts

From project root:

- npm run dev: runs frontend dev server
- npm run build: builds frontend for production
- npm run preview: previews frontend production build

## Troubleshooting

### Error: Missing script dev

Run commands from project root where package.json exists.

### Backend upload fails

Check:

- backend server is running on port 5000
- CSV includes transaction data columns (for example transaction_amount and transaction_timestamp)
- Python dependencies installed in the same environment used to run app.py

### CORS or network error from frontend

Ensure backend is running at http://127.0.0.1:5000.

If needed, configure frontend API base URL using VITE_API_BASE_URL.

## Notes

- The first push included frontend node_modules in git history. Consider adding a .gitignore and cleaning tracked dependencies in a follow-up commit.
- For large datasets, add backend batching/async processing for faster inference.
