import pandas as pd

import pandas as pd
import requests
import numpy as np
from collections import defaultdict
from src.data_loaders.helpers import get_vote_type_groups, chamber_to_value
from src.data_loaders.utils import parquet_cache
from src.data_loaders.data_paths import (
    ROLLCALL_QUERY,
    ROLLCALL_CLEANED,
    ROLLCALL_CRS_POLICY,
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
    return df.set_index(["date", "congress", "session", "bill_number", "rollnumber"])


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
    policies = policy_areas.unique()
    one_hot_policy_areas = pd.get_dummies(policy_areas).astype(int)
    one_hot_policy_areas = one_hot_policy_areas.rename(
        columns={c: "crs_policy_area_" + c for c in one_hot_policy_areas.columns}
    )
    df = df.drop(columns="crs_policy_area").join(one_hot_policy_areas)
    return df[
        [
            "chamber",
            "vote_passed",
            *[c for c in df.columns if c.startswith("vote_type_")],
            *[c for c in df.columns if c.startswith("crs_policy_area_")],
        ]
    ]
