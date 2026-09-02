from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

@dataclass
class DatasetSplits:
  train:pd.DataFrame
  validation:pd.DataFrame
  test:pd.DataFrame
  
  
def split_dataset(df:pd.DataFrame,test_size:float=0.15,
                  validation_size:float=0.15,random_state=42)->DatasetSplits:
  if test_size <=0 or validation_size <=0:
    raise ValueError(
            "test_size and validation_size must be greater than 0"
        )
  if test_size + validation_size >=1:
    raise ValueError(
            "test_size + validation_size must be less than 1"
        )
    
  train_val_df,test_df = train_test_split(
    df,
    test_size=test_size,
    random_state=random_state,
    stratify=df["label"]
  )
  relative_validation_size = (
    validation_size / (1.0-test_size)
  )  
  train_df,validation_df = train_test_split(
    train_val_df,
    test_size= relative_validation_size,
    stratify=train_val_df["label"]
  )
  
  train_df = train_df.reset_index(drop=True)
  validation_df = validation_df.reset_index(drop=True)
  test_df = test_df.reset_index(drop=True)
  
  return DatasetSplits(
    train=train_df,
    validation=validation_df,
    test=test_df
  )
  

def save_splits(
  splits: DatasetSplits,
  output_dir : str|Path
)->None:
  
  output_dir = Path(output_dir)
  output_dir.mkdir(
    parents=True,
    exist_ok=True,
  )
  splits.train.to_csv(
    output_dir/"train.csv",
    index=False
  )
  splits.validation.to_csv(
    output_dir/"validation.csv",
    index=False
  )
  splits.test.to_csv(
        output_dir / "test.csv",
        index=False,
    )
