# augmentation.py - Optional untuk tambah data
import pandas as pd
import random

def augment_data(df):
    """Augment data untuk kelas dengan sampel sedikit"""
    augmented_rows = []
    
    for intent in df['intent'].unique():
        intent_data = df[df['intent'] == intent]
        
        # Jika kelas punya <= 2 sampel, augment
        if len(intent_data) <= 2:
            print(f"🔄 Augmenting data untuk: {intent}")
            
            for _, row in intent_data.iterrows():
                pattern = row['pattern']
                
                # Simple augmentation: synonym replacement
                augmentations = [
                    pattern,
                    pattern.replace('cara', 'bagaimana'),
                    pattern.replace('buat', 'bikin'),
                    pattern.replace('dimana', 'di mana'),
                    pattern + ' ya',
                    'tolong ' + pattern
                ]
                
                for aug_pattern in augmentations:
                    if aug_pattern != pattern:  # Jangan duplicate original
                        new_row = row.copy()
                        new_row['pattern'] = aug_pattern
                        augmented_rows.append(new_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    final_df = pd.concat([df, augmented_df], ignore_index=True)
    
    print(f"📊 Data augmented: {len(df)} -> {len(final_df)} rows")
    return final_df

# Usage:
# df = pd.read_csv('dataset/dataset_training.csv')
# df_augmented = augment_data(df)
# df_augmented.to_csv('dataset/dataset_augmented.csv', index=False)