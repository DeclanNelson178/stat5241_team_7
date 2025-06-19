def get_vote_type_groups():
    return {
        # this is a vote in favor of the amendment
        "amend": [
            "on agreeing to the amendment",
            "on the amendment",
            "on agreeing to the amendments en bloc",
            "on agreeing to the substitute amendment",
            "on agreeing to the amendments",
            "on agreeing to the amendment, as amended",
            "on agreeing to the senate amendment",
            "on agreeing to the amendment, as modified",
            "whether the amendment is germane",
        ],
        # this is a passage of the bill
        "pass": [
            "on passage",
            "on passage of the bill",
            "passage, objections of the president notwithstanding",
            "passage, objections of the president not withstanding",
            "passage, objections of the president to the contrary notwithstanding",
            "passage, objections ofthe president notwithstanding",
            "passage, objection of the president notwithstanding",
        ],
        # tabling is way to kill a bill -- a vote yes here is equivalent to voting no on the bill
        "table": [
            "on the motion to table",
            "table motion to reconsider",
        ],
        # suspending rules means that the bill is fast tracked to a vote -- it is essentially a vote in favor of the bill
        "suspend": [
            "on motion to suspend the rules and pass",
            "on motion to suspend the rules and pass, as amended",
            "suspend the rules and pass, as amended",
            "suspend the rules and pass",
            "suspend the rules and pass as amended",
            "suspend the rules and agree to senate amendment",
            "motion to suspend the rules and pass, as amended",
            "suspend the rules and agree to senate amendments",
            "suspend rules and pass, as amended",
            "suspend rules and passas amended",
            "motion to suspend the rules and pass",
            "suspend the rules and concur in the senate amendment",
            "suspend the rules and agree to the senate amendment",
            "suspend the rules and agree to conference report",
            "on motion to suspend rules and pass",
            "on motion to suspend rules and pass, as amended",
        ],
        # requesting changes to the bill -- tatic used to stall or kill the bill
        "recommit": [
            "on motion to recommit with instructions",
            "on motion to recommit",
            "on the motion to recommit",
            "on motion to commit with instructions",
            "on motion to recommit the conference report",
            "recommit conference report with instructions",
            "recommit the conference report with instructions",
            "on motion to recommit conference report with instructions",
            "motion to recommit conference report with instructions",
            "on motion to commit",
        ],
        # end debate a proceed to the vote -- often indicates a willingness to vote yes
        "cloture": [
            "on the cloture motion",
            "on cloture on the motion to proceed",
        ],
        # agreeing to the bill as passed by the other house -- voting yes
        "conference": [
            "on agreeing to the conference report",
            "on the conference report",
            "on motion to suspend the rules and agree to the conference report",
        ],
        # enact the bill regardless of the presidential veto -- strongly voting yes
        "veto": ["on overriding the veto", "on presidential veto"],
        # accept changes made by the senate -- voting yes
        "concur": [
            "on motion to suspend the rules and concur in the senate amendment",
            "on motion to concur in the senate amendment",
            "on motion to concur in the senate amendment with an amendment",
            "on motion to suspend the rules and concur in the senate amendments",
            "on motion to concur in the senate amendments",
            "on motion to concur in senate amendments",
            "on motion to agree to the senate amendment",
            "on motion to concur in the senate adt to the house adt to the senate adt",
            "on motion to concur in the senate amdt to the house amdt to the senate amdt",
            "agree to senate amendments",
        ],
    }


def chamber_to_value(chamber: str):
    """Binary encoding of house and senate values"""
    return {
        "house": 0,
        "senate": 1,
    }[chamber]


def value_to_chamber(val: int):
    """Binary encoding of house and senate values"""
    return {0: "house", 1: "senate"}[val]


def is_democrat(party_code: int) -> int:
    return party_code == 100


def is_republican(party_code: int) -> int:
    return party_code == 200
