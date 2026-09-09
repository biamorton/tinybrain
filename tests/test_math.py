from tinybrain.skills.math_skill import SafeMathSkill


def test_math_skill():
    skill = SafeMathSkill()
    assert skill.run("37 * 14") == "518"
    assert skill.run("(18 + 7) * 3") == "75"
