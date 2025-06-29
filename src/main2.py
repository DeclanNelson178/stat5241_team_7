from src.data_loaders.rollcall import (
    get_individual_votes_with_party_enriched_lobby,
    get_training_data_v6,
    get_training_data_pass_only,
    get_individual_votes_with_party_enriched,
)

if __name__ == "__main__":
    df = get_individual_votes_with_party_enriched_lobby()
    breakpoint()
    print(df)
