"""``XLA_FLAGS`` parsed and merged by flag name, with conflicting values refused."""

from __future__ import annotations

import pytest

from substrax.runtime import (
    merge_xla_flags,
    parse_xla_flags,
    RuntimeConfigurationError,
    XlaFlagConflictError,
)


DEVICE_COUNT = "--xla_force_host_platform_device_count"


class TestParseXlaFlags:
    def test_an_empty_value_has_no_flags(self) -> None:
        assert parse_xla_flags("") == {}

    def test_any_whitespace_separates_flags(self) -> None:
        assert parse_xla_flags("  --a=1 \t --b=2\n --c=3") == {"--a": "1", "--b": "2", "--c": "3"}

    def test_a_flag_without_a_value_keeps_no_value(self) -> None:
        assert parse_xla_flags("--xla_dump_to_stdout") == {"--xla_dump_to_stdout": None}

    def test_a_value_may_contain_an_equals_sign(self) -> None:
        assert parse_xla_flags("--a=b=c") == {"--a": "b=c"}

    def test_a_token_that_is_not_a_flag_raises(self) -> None:
        with pytest.raises(ValueError, match="xla_gpu_autotune_level=2"):
            parse_xla_flags("--a=1 xla_gpu_autotune_level=2")

    def test_one_flag_given_twice_with_one_value_is_one_flag(self) -> None:
        assert parse_xla_flags("--a=1 --b=2 --a=1") == {"--a": "1", "--b": "2"}

    def test_one_flag_given_twice_with_different_values_raises(self) -> None:
        with pytest.raises(XlaFlagConflictError, match=DEVICE_COUNT):
            parse_xla_flags(f"{DEVICE_COUNT}=8 {DEVICE_COUNT}=1")


class TestMergeXlaFlags:
    def test_a_new_flag_follows_the_existing_ones(self) -> None:
        assert merge_xla_flags("--a=1 --b=2", ["--c=3"]) == "--a=1 --b=2 --c=3"

    def test_a_requested_flag_already_set_to_the_same_value_changes_nothing(self) -> None:
        assert merge_xla_flags("--a=1 --b=2 --c=3", ["--b=2"]) == "--a=1 --b=2 --c=3"

    def test_untouched_flags_keep_their_order(self) -> None:
        assert (
            merge_xla_flags("--c=3 --a=1 --b=2", ["--d=4", "--e"]) == "--c=3 --a=1 --b=2 --d=4 --e"
        )

    def test_nothing_requested_normalises_the_spacing(self) -> None:
        assert merge_xla_flags("  --a=1   --b ", []) == "--a=1 --b"

    def test_an_existing_device_count_is_never_overridden_by_a_request(self) -> None:
        with pytest.raises(XlaFlagConflictError, match=DEVICE_COUNT):
            merge_xla_flags(f"--a=1 {DEVICE_COUNT}=8", [f"{DEVICE_COUNT}=1"])

    def test_the_conflict_names_the_flag_and_both_values(self) -> None:
        with pytest.raises(XlaFlagConflictError) as caught:
            merge_xla_flags("--a=1", ["--a=2"])

        assert (caught.value.flag, caught.value.existing, caught.value.requested) == (
            "--a",
            "1",
            "2",
        )

    def test_a_flag_and_the_same_flag_with_a_value_conflict(self) -> None:
        with pytest.raises(XlaFlagConflictError, match="--a"):
            merge_xla_flags("--a", ["--a=true"])

    def test_a_requested_flag_holding_whitespace_raises(self) -> None:
        """XLA splits ``XLA_FLAGS`` on whitespace, so such a flag would arrive as two tokens."""
        with pytest.raises(ValueError, match="xla_dump_to"):
            merge_xla_flags("--a=1", ["--xla_dump_to=/tmp/two words"])

    def test_a_conflict_is_a_runtime_configuration_error(self) -> None:
        assert issubclass(XlaFlagConflictError, RuntimeConfigurationError)
        assert issubclass(RuntimeConfigurationError, RuntimeError)
