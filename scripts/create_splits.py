from src.data.split import (
  save_splits,
  split_dataset
)

from src.data.validation import(
  load_dataset,
  validate_dataset
)

DATA_PATH = "data/raw/tickets.csv"
OUTPUT_DIR = "data/splits"

def main()->None:
  df = load_dataset(DATA_PATH)
  
  validate_dataset(df)
  
  splits = split_dataset(
    df=df,
    test_size=0.15,
    validation_size=0.15,
    random_state=42,
  )
  save_splits(
    splits=splits,
    output_dir=OUTPUT_DIR
  )
  print("Dataset split successfully.")
  print(f"Train:      {len(splits.train)}")
  print(f"Validation: {len(splits.validation)}")
  print(f"Test:       {len(splits.test)}")
  
if __name__ =="__main__":
  main()