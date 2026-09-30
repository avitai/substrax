# The pull request whose merge produced the commit $sha, from GitHub's list of pull requests
# associated with that commit (GET /repos/{owner}/{repo}/commits/{sha}/pulls). A squash, rebase or
# merge commit is a pull request's `merge_commit_sha` once it merged; an open pull request lists a
# test merge there, so only merged ones count. Prints nothing when no merged pull request made it.
[.[] | select(.merged_at != null and .merge_commit_sha == $sha) | .number] | first // empty
