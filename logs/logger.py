import os
import pandas as pd

LOG_FILE_PATH = os.path.join("logs", "experiment_history.csv")
EXACT_COLUMNS = ["File Name", "Model Used", "Prediction", "Confidence (in %)"]

# 1. Ensure the directory exists
os.makedirs("logs", exist_ok=True)

# 2. Create a fresh CSV if it doesn't exist
if not os.path.exists(LOG_FILE_PATH):
    empty_df = pd.DataFrame(columns=EXACT_COLUMNS)
    empty_df.to_csv(LOG_FILE_PATH, index=False)