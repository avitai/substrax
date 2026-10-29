# The checks of a pull request that are not proven, read from the response of checks.graphql: every
# check whose conclusion (a check run) or state (a commit status) is anything but SUCCESS, SKIPPED or
# NEUTRAL. A check still running has no conclusion, and a cancelled or action-required one never
# passed: each keeps the merge tested. A check past the first page was not read, and a pull request
# with no check proved nothing, so each of those counts as one more.
.data.repository.pullRequest.commits.nodes[0].commit.statusCheckRollup as $rollup
| ($rollup.contexts.nodes // []) as $checks
| [$checks[]
   | ((.conclusion // .state) // "")
   | select(. != "SUCCESS" and . != "SKIPPED" and . != "NEUTRAL")]
| length
  + (if $rollup.contexts.pageInfo.hasNextPage then 1 else 0 end)
  + (if $checks == [] then 1 else 0 end)
