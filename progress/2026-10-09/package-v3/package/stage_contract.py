"""One stage namespace shared by the launcher, worker and supervisor."""

REUSE_MODE = "matched_continuation_fusion_reuse_v3"
STAGE_ORDER = (
    "input_admission",
    "paired_collection",
    "world_training_and_restore",
    "offline_prediction",
    "conditional_policy_training",
    "conditional_task_confirmation",
    "analysis_and_settlement",
    "settlement_and_verified_export",
)


def initial_stage(request):
    order = request.get("stage_order")
    if order is None:
        # Historical contracts remain readable by shared protocol tests.
        stage = "staging_and_zero_step_gate"
    else:
        if (not isinstance(order, list) or not order or len(set(order)) != len(order)
                or set(order) != set(request["stages"])):
            raise ValueError("STAGE_ORDER_RESOURCE_CONTRACT_MISMATCH")
        stage = order[0]
    if stage not in request["stages"]:
        raise ValueError("INITIAL_STAGE_MISSING_FROM_RESOURCE_REQUEST")
    return stage


def is_fusion_mode(matrix):
    return matrix.get("research_mode") in {
        "matched_continuation_fusion", "matched_continuation_fusion_opportunity_v2", REUSE_MODE,
    }
