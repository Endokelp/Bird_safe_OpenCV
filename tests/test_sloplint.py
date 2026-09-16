from tools.sloplint import ALLOW_MARKER, find_violations

EM = chr(0x2014)


def test_flags_em_dash():
    assert find_violations(f"a {EM} b", "f.md")


def test_flags_banned_word_case_insensitively():
    assert find_violations("We " + "Lever" + "age this", "f.md")


def test_flags_not_just_but_construction():
    assert find_violations("not " + "just fast, but correct", "f.md")


def test_flags_todo_without_owner():
    assert find_violations("# TO" + "DO fix this", "f.py")


def test_allows_todo_with_owner_and_reason():
    assert not find_violations("# TO" + "DO(venni): retune after the Oct 2 captures", "f.py")


def test_allow_marker_exempts_a_line_that_quotes_the_rule():
    assert not find_violations(f"Avoid the {EM} character. {ALLOW_MARKER}", "AGENTS.md")


def test_clean_text_passes():
    assert not find_violations("The gap measured 61 mm against a 50.8 mm threshold.", "f.md")


def test_hyphen_is_not_an_em_dash():
    assert not find_violations("multi-frame fusion", "f.md")
