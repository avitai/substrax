# The checks of a pull request that are not proven: every check whose conclusion (a check run) or
# state (a commit status) is anything but SUCCESS, SKIPPED or NEUTRAL. A check still running has an
# empty conclusion, and a cancelled or action-required one never passed: each keeps the merge tested.
[.statusCheckRollup[]
  | ((.conclusion // .state) // "")
  | select(. != "SUCCESS" and . != "SKIPPED" and . != "NEUTRAL")]
| length
