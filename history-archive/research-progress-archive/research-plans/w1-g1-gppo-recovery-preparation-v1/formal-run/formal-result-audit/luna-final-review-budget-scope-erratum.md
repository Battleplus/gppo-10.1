# Budget Scope Erratum

This erratum corrects the budget interpretation in the initial Luna final review. The review files have also been updated to carry the corrected values.

The superseded statement was: “cumulative launch wall is 4,805.325 seconds against the 4,670-second global cap.” This compared old-plus-new measured wall with the cap for the new attempt and incorrectly described a wall overrun.

The contract in `package/cumulative-resource-contract.json` gives the new attempt a 4,670-second wall cap. Its cumulative disclosure envelope is old measured wall plus the new cap: 3,629.3990414 + 4,670 = 8,299.3990414 seconds. The new attempt measured 1,175.9259505 seconds and passed its wall check. Old-plus-new measured wall is 4,805.3249919 seconds, within the cumulative envelope by 3,494.0740495 seconds.

The overall `full_resource_acceptance=false` status remains. It reflects the disclosed resource measurement gaps and unresolved old-attempt settlement, including one reserved update whose completion remains unknown. It does not indicate that the new attempt exceeded its wall cap.

No raw evidence or package contract was modified. Only the Luna review JSON, Luna review Markdown, and this erratum were written.
