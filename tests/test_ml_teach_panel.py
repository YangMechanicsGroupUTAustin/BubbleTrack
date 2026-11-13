"""ML assist's teaching panel on its own: cards, the question, painting folded
away, the run lock (no controller)."""

from __future__ import annotations

from bubbletrack.ui.ml_teach_panel import CONFIRMED, LOOKS_RIGHT, LOOKS_WRONG, CardView, MLTeachPanel


def _panel(qapp):
    panel = MLTeachPanel()
    panel.show()
    qapp.processEvents()
    return panel


def test_cards_ask_until_confirmed(qapp):
    panel = _panel(qapp)
    panel.set_cards([CardView(4, "Frame 5 · R = 30 px"), CardView(40, "Frame 41 · R = 12 px")])
    assert panel.card_frames() == [4, 40] and not panel._question.isHidden()
    card = panel._cards[0]
    assert card.right.text() == LOOKS_RIGHT and card.wrong.text() == LOOKS_WRONG
    assert card.status.isHidden() and not card.right.isHidden()
    asked = []
    panel.confirm_requested.connect(lambda f: asked.append(("right", f)))
    panel.reject_requested.connect(lambda f: asked.append(("wrong", f)))
    panel._cards[0].right.click()
    panel._cards[1].wrong.click()
    assert asked == [("right", 4), ("wrong", 40)]
    panel.set_cards([CardView(4, "Frame 5 · R = 30 px", confirmed=True),
                     CardView(40, "Frame 41 · R = 12 px", confirmed=True)])
    assert panel.card_confirmed(4) and card.status.text() == CONFIRMED and card.right.isHidden()
    assert panel._question.isHidden()                       # nothing left to judge
    panel.set_cards([None, None])
    assert panel.card_frames() == [] and panel._cards[0].isHidden()


def test_painting_is_folded_and_reports_its_toggle(qapp):
    panel = _panel(qapp)
    seen = []
    panel.paint_toggled.connect(seen.append)
    assert not panel.painting_open() and panel._paint_body.isHidden()
    panel.set_painting(True)
    assert panel.painting_open() and not panel._paint_body.isHidden() and seen == [True]
    panel.paint_toggle.click()
    assert not panel.painting_open() and seen == [True, False]


def test_a_run_locks_judging_and_training_and_shows_stop(qapp):
    panel = _panel(qapp)
    panel.set_train(True, "go")
    panel.set_cards([CardView(4, "Frame 5")])
    panel.set_running(True)
    assert not panel.train_button.isEnabled() and not panel._cards[0].right.isEnabled()
    panel.set_cards([CardView(4, "Frame 5")])               # a refresh during the run
    assert not panel._cards[0].right.isEnabled()
    assert not panel._progress.isHidden() and not panel._stop_btn.isHidden()
    panel.set_running(False)
    assert panel.train_button.isEnabled() and panel._progress.isHidden()
