from src.data_loaders.rollcall import (
    get_individual_votes_with_party_enriched_lobby,
    get_training_data_v6,
)

if __name__ == "__main__":
    df = get_training_data_v6()[-1]
    print(df.columns)
    breakpoint()
    print(df)
