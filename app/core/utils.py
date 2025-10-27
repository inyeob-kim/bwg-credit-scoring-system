import time
import numpy as np
from app.core.config import PDO, S0, O0




def prob_to_score(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    odds = (1 - p) / p
    factor = PDO / np.log(2)
    offset = S0 - factor * np.log(O0)
    return offset + factor * np.log(odds)




def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")