import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('../../data/processed/spectrogram_final.csv')


def histograme_occurence(nbr_occurences = 100):

    accent_counts = df['accents'].value_counts()
    filtered_counts = accent_counts[accent_counts >= nbr_occurences]

    plt.figure(figsize=(10, 6))
    filtered_counts.plot(kind='bar', color='mediumseagreen')
    plt.title(f'Accents avec au moins {nbr_occurences} occurrences')
    plt.xlabel('Accent')
    plt.ylabel('Nombre d\'occurrences')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()