from functools import lru_cache
from typing import List, Tuple
import pandas as pd

import pandas as pd
import requests
import numpy as np
from collections import defaultdict
from src.data_loaders.helpers import (
    get_vote_type_groups,
    chamber_to_value,
    is_democrat,
    is_republican,
)
from src.data_loaders.utils import parquet_cache
from src.data_loaders.data_paths import (
    INDIVIDUAL_VOTES_V1,
    ROLLCALL_CLEANED,
    ROLLCALL_CRS_POLICY,
    RAW_INDIVIDUAL_VOTES,
    RAW_PARTY_MEMBERSHIP,
    TRAINING_DATA_V2,
    VOTE_WITH_PARTY,
)


def query_rollcall_data():
    # query the raw data source
    url = f"https://voteview.com/static/data/out/rollcalls/HSall_rollcalls.json"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
    else:
        print("Error:", response.status_code)
    return pd.DataFrame.from_records(data)


@parquet_cache(ROLLCALL_CLEANED)
def get_cleaned_rollcall_data():
    """Clean rollcall data"""
    df = query_rollcall_data().drop(
        columns=["congress_url", "clerk_rollnumber", "source_documents"]
    )

    # baseline cleaning -- require an indentifier to work with
    df = df.dropna(
        subset=[
            "congress",
            "bill_number",
            "rollnumber",
            "vote_question",
            "vote_desc",
            "crs_policy_area",
        ]
    )
    df["congress"] = df["congress"].astype(int)
    df["bill_number"] = df["bill_number"].astype(str).str.lower().str.strip()
    df["vote_question"] = df["vote_question"].astype(str).str.lower().str.strip()
    df["rollnumber"] = df["rollnumber"].astype(int)
    df["chamber"] = df["chamber"].astype(str).str.lower().str.strip()
    df["chamber"] = df["chamber"].apply(chamber_to_value)

    # classify bills
    df["bill_type"] = df["bill_number"].str.replace(r"\d+", "", regex=True)

    # these are canonical bills, the other bill types are resolutions, treaties, procedural and nomations
    bill_types = ["hr", "s", "hj", "hjr", "hjres", "hjre", "sj", "sjr", "sjres", "sjre"]
    df = df.loc[df["bill_type"].isin(bill_types)]

    # Remove any rows that are unusable -- you need to have a bill, a vote, and a location of the vote
    df = df.dropna(subset=["yea_count", "nay_count", "chamber"], how="any")

    # Filter to valid vote types
    vote_type_groups = get_vote_type_groups()
    valid_vote_questions = []
    for vote_questions in vote_type_groups.values():
        valid_vote_questions.extend(vote_questions)

    vote_question_to_vote_type = defaultdict(None)
    for vote_type, vote_questions in vote_type_groups.items():
        vote_question_to_vote_type.update(
            {vote_question: vote_type for vote_question in vote_questions}
        )

    # replace vote question with vote type
    df = df.loc[df["vote_question"].isin(valid_vote_questions)]
    df["vote_type"] = df["vote_question"].replace(vote_question_to_vote_type)
    df = df.drop(columns="vote_question")

    # one hot encoding of vote type
    dummies_vote_type = pd.get_dummies(df["vote_type"]).astype(int)
    dummies_vote_type = dummies_vote_type.rename(
        columns={c: "vote_type_" + c for c in dummies_vote_type.columns}
    )
    df = df.drop(columns="vote_type").join(dummies_vote_type)

    # add if the vote passed
    vote_results = df["vote_result"].unique()
    passed_results = [
        res
        for res in vote_results
        if any(
            res_type in res.lower()
            for res_type in ["passed", "agreed", "overridden", "germane"]
        )
    ]

    df["vote_passed"] = df["vote_result"].isin(passed_results).fillna(False).astype(int)
    df = df.drop(columns="vote_result")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(
        ["date", "congress", "session", "chamber", "bill_number", "rollnumber"]
    )


@parquet_cache(ROLLCALL_CRS_POLICY)
def get_rollcall_data_crs_policy_areas():
    """
    Fetch clean rollcall data and create a one hot encoding for crs_policy_areas

    This data is cleaned. We are only return integer values. 0 indicates exclusion, 1 indicates
    inclusion.
    """
    df = get_cleaned_rollcall_data()

    # need to handle codes and policy areas
    # first pass let's just use the crs_policy_area
    policy_areas = (
        df["crs_policy_area"]
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(",", "")
        .str.replace(".", "")
    )
    one_hot_policy_areas = pd.get_dummies(policy_areas).astype(int)
    one_hot_policy_areas = one_hot_policy_areas.rename(
        columns={c: "crs_policy_area_" + c for c in one_hot_policy_areas.columns}
    )
    df = df.drop(columns="crs_policy_area").join(one_hot_policy_areas)
    return df[
        [
            "vote_passed",
            *[c for c in df.columns if c.startswith("vote_type_")],
            *[c for c in df.columns if c.startswith("crs_policy_area_")],
        ]
    ]


def get_raw_individual_votes():
    return pd.read_parquet(RAW_INDIVIDUAL_VOTES).set_index(
        ["congress", "chamber", "rollnumber", "icpsr"]
    )[["vote_for", "vote_against"]]


@parquet_cache(INDIVIDUAL_VOTES_V1)
def get_individual_votes(vote_for_only: bool):
    ind_df = get_raw_individual_votes()
    bill_df = get_rollcall_data_crs_policy_areas().reset_index(["date", "bill_number"])
    df = bill_df.join(ind_df)
    if vote_for_only:
        # remove anyone who abstained from the vote
        df = df.loc[(df["vote_for"] == 1) | (df["vote_against"] == 1)]
        df = df.drop(columns=["vote_against"])

    return df


@lru_cache(maxsize=1)
def get_training_data_v1() -> Tuple[str, List[str], pd.DataFrame]:
    """
    Simple training data with the following predictors:
    - personal id
    - Congressional number
    - Chamber
    - Session
    - Vote type: ammend, cloture, concur, pass, recommit, suspend, table, veto
    - Crs policy areas: eduction, etc.

    Response variable is: Vote for the bill
    """
    df = get_individual_votes(vote_for_only=True).reset_index()
    target = "vote_for"
    features = [
        "icpsr",
        "congress",
        "chamber",
        "session",
        *[c for c in df.columns if c.startswith("vote_type_")],
        *[c for c in df.columns if c.startswith("crs_policy_area")],
    ]
    df = df[[target, *features]]
    return target, features, df


def get_raw_party_membership():
    return pd.read_csv(RAW_PARTY_MEMBERSHIP)


@parquet_cache(VOTE_WITH_PARTY)
def get_individual_votes_with_party() -> pd.DataFrame:
    df = get_individual_votes(vote_for_only=True)
    party_df = get_raw_party_membership()
    party_df["d"] = party_df["party_code"].apply(is_democrat)
    party_df["r"] = party_df["party_code"].apply(is_republican)
    return df.join(party_df.set_index(["congress", "icpsr"])[["d", "r"]])


@lru_cache(maxsize=1)
def get_trainig_data_v2():
    df = get_individual_votes_with_party().reset_index()
    target = "vote_for"
    features = [
        "icpsr",
        "congress",
        "chamber",
        "session",
        "d",
        "r",
        *[c for c in df.columns if c.startswith("vote_type_")],
        *[c for c in df.columns if c.startswith("crs_policy_area")],
    ]
    df = df[[target, *features]]
    return target, features, df
