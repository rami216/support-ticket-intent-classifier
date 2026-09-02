from src.data.validation import (
    DataValidationError,
    load_dataset,
    validate_dataset,
)

DATA_PATH = "data/raw/tickets.csv"

def main()->None:
  try:
    df = load_dataset(DATA_PATH)
    validate_dataset(df)
    print("Dataset validation passed.")
    print(f"Rows:{len(df)}")
  except DataValidationError as exc:
    print("Dataset validation failed.")
    print(exc)
    
    raise SystemExit(1)
  
if __name__ == "__main__":
  main()