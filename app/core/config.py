import os

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "./artifacts")
PDO = int(os.environ.get("PDO", 50))
S0 = int(os.environ.get("S0", 600))
O0 = int(os.environ.get("O0", 20))