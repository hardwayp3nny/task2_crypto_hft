# FinRL Task 2 - AlphaSeek Crypto

## Workflow

### 1. Environment Setup

1. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # For Windows
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download and place the dataset:**

   - Place the `BTC_1sec.csv` (1-second interval BTC order book data) in the `./data` directory.

---

### 2. Data Preprocessing & Label Generation

Run the following script to preprocess the data and generate labels:

```bash
python seq_data.py
```

---

### 3. Factor Model Training

Train your factor-based models using:

```bash
python seq_run.py
```

---

### 4. Reinforcement Learning Training

Use the following command to train reinforcement learning agents:

```bash
python erl_run.py
```

---

### 5. Model Evaluation

Evaluate the model's performance by running:

```bash
python task2_eval.py
```
