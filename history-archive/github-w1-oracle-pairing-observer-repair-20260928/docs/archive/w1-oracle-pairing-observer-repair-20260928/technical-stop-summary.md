# Oracle Technical Stop Summary

The attempt `w1-action-consequence-oracle-ceiling-v1-once` stopped in its first fixed unit after 90 environment steps, one reset, 84 public-rule decisions, and seven completed branches. No model or training calls occurred.

The original verifier reported 149 mismatches among 983 common coarse identities. The offline audit reproduced those counts and traced every mismatch:

- 135 telemetry mismatches compared complete high-level event histories. The primitive send fate matched; branches ended at different times, so later `received` rows differed.
- 8 command and 6 ACK mismatches used coarse `command_id` grouping and merged allocation, renewal, or ACK events with different semantic identities.

The saved logs do not contain complete primitive input addresses for the affected command/ACK calls. Those fields were not reconstructed as if observed. The seven branches therefore remain invalid for oracle-ceiling or materiality analysis.

The repaired observer records future primitive calls at their actual boundary and compares exact identities, parameters, multiplicity, and returned fate. This is a prepared technical fix only. No dynamic validation was run under this archive task.

The completed repaired-runtime fair rerun remains valid under its frozen interpretation because it did not use the oracle verifier. Its negative task result, historical resource costs, and the 10 ms decision-cost failure remain unchanged.
