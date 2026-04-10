"""
Unit tests for integrations.social.game_ai — server-side AI move generators.

Covers:
  - Difficulty → error rate mapping
  - Per-engine move generation (trivia, opentdb, word_scramble,
    word_search, sudoku)
  - Ground-truth preservation at 'hard' difficulty (near-perfect play)
  - Error injection at 'easy' difficulty (statistical test with
    deterministic seed)
  - Client-authoritative engine rejection (boardgame, phaser)
  - Unknown engine raises ValueError
  - Move shape matches each game type's apply_move() expectation

No DB or network — pure functions only.
"""

import copy
import random

import pytest

from integrations.social.game_ai import (
    CLIENT_AUTHORITATIVE_ENGINES,
    DIFFICULTY_ERROR_RATE,
    MultipleChoiceTriviaAI,
    OpenTDBTriviaAI,
    SudokuAI,
    TriviaAI,
    WordScrambleAI,
    WordSearchAI,
    _error_rate,
    generate_ai_move,
    get_game_ai,
    is_client_authoritative,
)


# ─── Difficulty mapping ──────────────────────────────────────────────

class TestDifficultyMapping:
    def test_easy_is_harder_than_medium(self):
        assert DIFFICULTY_ERROR_RATE['easy'] > DIFFICULTY_ERROR_RATE['medium']

    def test_medium_is_harder_than_hard(self):
        assert DIFFICULTY_ERROR_RATE['medium'] > DIFFICULTY_ERROR_RATE['hard']

    def test_error_rate_defaults_to_medium_on_unknown(self):
        assert _error_rate('nonsense') == DIFFICULTY_ERROR_RATE['medium']


# ─── TriviaAI ────────────────────────────────────────────────────────

class TestTriviaAI:
    def _state(self):
        return {
            'current_question_idx': 0,
            'questions': [
                {'q': 'Capital of France?', 'a': 'paris',
                 'options': ['paris', 'london', 'berlin', 'madrid']},
            ],
        }

    def test_hard_picks_correct_answer(self):
        random.seed(12345)
        ai = TriviaAI()
        state = self._state()
        # Hard difficulty: error rate ~4%. Over 50 runs, should be
        # correct the vast majority of the time.
        correct_count = sum(
            1 for _ in range(50)
            if ai.generate_move(state, 'ai', 'hard')['answer'] == 'paris'
        )
        assert correct_count >= 42  # allow up to 8 wrong in 50

    def test_easy_picks_wrong_sometimes(self):
        random.seed(54321)
        ai = TriviaAI()
        state = self._state()
        # Easy: error rate ~45%. Over 100 runs, should see at least
        # 10 wrong answers (very loose lower bound).
        wrong_count = sum(
            1 for _ in range(100)
            if ai.generate_move(state, 'ai', 'easy')['answer'] != 'paris'
        )
        assert wrong_count >= 10

    def test_move_shape_matches_trivia_apply_move(self):
        """TriviaGame.apply_move reads move_data['answer']."""
        random.seed(1)
        move = TriviaAI().generate_move(self._state(), 'ai', 'hard')
        assert 'answer' in move
        assert isinstance(move['answer'], str)

    def test_raises_when_no_active_question(self):
        ai = TriviaAI()
        with pytest.raises(ValueError, match='No active question'):
            ai.generate_move({'current_question_idx': 5, 'questions': []},
                             'ai', 'medium')


# ─── OpenTDBTriviaAI ─────────────────────────────────────────────────

class TestOpenTDBTriviaAI:
    """OpenTDBTriviaGame normalizes questions to the same 'a'/'options'
    shape as TriviaGame (see game_types_extended.py:75-89). So the AI
    handler is literally the same class — these tests verify the alias
    resolves and the shape assumption holds.
    """

    def _state(self):
        # Matches what OpenTDBTriviaGame._fetch_questions actually stores:
        # unescaped question text, shuffled options, 'a' + 'options' keys.
        return {
            'current_question_idx': 0,
            'questions': [{
                'q': 'Which planet?',
                'a': 'Mars',
                'options': ['Venus', 'Jupiter', 'Mars', 'Saturn'],
                'difficulty': 'medium',
            }],
        }

    def test_opentdb_and_trivia_are_same_class(self):
        """DRY: single handler serves both trivia engines."""
        assert OpenTDBTriviaAI is TriviaAI
        assert OpenTDBTriviaAI is MultipleChoiceTriviaAI

    def test_hard_returns_correct_answer(self):
        random.seed(7)
        ai = OpenTDBTriviaAI()
        correct_count = sum(
            1 for _ in range(50)
            if ai.generate_move(self._state(), 'ai', 'hard')['answer'] == 'Mars'
        )
        assert correct_count >= 42

    def test_wrong_answer_comes_from_options(self):
        random.seed(99)
        ai = OpenTDBTriviaAI()
        state = self._state()
        valid = {'Mars', 'Venus', 'Jupiter', 'Saturn'}
        for _ in range(20):
            move = ai.generate_move(state, 'ai', 'easy')
            assert move['answer'] in valid

    def test_single_option_question_never_errors(self):
        """When options has only 1 entry, the AI has no wrong choice
        to pick — should always return the correct answer regardless
        of difficulty (no crash, no empty-string leak)."""
        state = {
            'current_question_idx': 0,
            'questions': [{'q': 'Q?', 'a': 'only-option',
                            'options': ['only-option']}],
        }
        random.seed(1)
        for _ in range(20):
            move = MultipleChoiceTriviaAI().generate_move(state, 'ai', 'easy')
            assert move['answer'] == 'only-option'

    def test_empty_options_never_errors(self):
        """Degenerate state: 'options' missing/empty. AI should return
        the correct answer (cannot pick a wrong option)."""
        state = {
            'current_question_idx': 0,
            'questions': [{'q': 'Q?', 'a': 'the-answer'}],
        }
        random.seed(1)
        for _ in range(10):
            move = MultipleChoiceTriviaAI().generate_move(state, 'ai', 'easy')
            assert move['answer'] == 'the-answer'


# ─── WordScrambleAI ──────────────────────────────────────────────────

class TestWordScrambleAI:
    def _state(self):
        return {
            'current_round_idx': 0,
            'rounds': [{
                'word': 'planet',
                'scrambled': 'telnap',
                'solved_by': None,
            }],
        }

    def test_hard_returns_correct_word(self):
        random.seed(42)
        ai = WordScrambleAI()
        correct_count = sum(
            1 for _ in range(50)
            if ai.generate_move(self._state(), 'ai', 'hard')['word'] == 'planet'
        )
        assert correct_count >= 42

    def test_move_includes_time_ms_for_score_bonus(self):
        """WordScrambleGame.apply_move reads move_data['time_ms']
        for the time bonus calculation."""
        random.seed(1)
        move = WordScrambleAI().generate_move(self._state(), 'ai', 'hard')
        assert 'time_ms' in move
        assert isinstance(move['time_ms'], int)
        assert move['time_ms'] > 0

    def test_hard_response_faster_than_easy(self):
        random.seed(1)
        fast = WordScrambleAI().generate_move(self._state(), 'ai', 'hard')
        random.seed(1)
        slow = WordScrambleAI().generate_move(self._state(), 'ai', 'easy')
        # 'hard' uses range 1500-5000; 'easy' uses 8000-15000 — hard max
        # is below easy min, so hard is always strictly faster.
        assert fast['time_ms'] < slow['time_ms']

    def test_raises_when_round_already_solved(self):
        state = {
            'current_round_idx': 0,
            'rounds': [{'word': 'planet', 'scrambled': 'telnap',
                        'solved_by': 'someone-else'}],
        }
        with pytest.raises(ValueError, match='already solved'):
            WordScrambleAI().generate_move(state, 'ai', 'medium')


# ─── WordSearchAI ────────────────────────────────────────────────────

class TestWordSearchAI:
    def test_hard_picks_unfound_words_almost_always(self):
        state = {
            'words_to_find': ['tiger', 'eagle', 'shark'],
            'found_words': {'tiger': 'alice'},
        }
        random.seed(1)
        # At hard (error rate ~4%), most picks should be unfound words.
        unfound_count = sum(
            1 for _ in range(50)
            if WordSearchAI().generate_move(state, 'ai', 'hard')['word']
            in {'eagle', 'shark'}
        )
        assert unfound_count >= 42

    def test_move_word_is_always_a_string(self):
        state = {
            'words_to_find': ['tiger', 'eagle', 'shark'],
            'found_words': {'tiger': 'alice'},
        }
        random.seed(1)
        for _ in range(20):
            move = WordSearchAI().generate_move(state, 'ai', 'medium')
            assert isinstance(move['word'], str)
            assert len(move['word']) > 0

    def test_raises_when_all_words_found(self):
        state = {
            'words_to_find': ['tiger', 'eagle'],
            'found_words': {'tiger': 'a', 'eagle': 'b'},
        }
        with pytest.raises(ValueError, match='No unfound'):
            WordSearchAI().generate_move(state, 'ai', 'medium')


# ─── SudokuAI ────────────────────────────────────────────────────────

class TestSudokuAI:
    def _state(self):
        solution = [[((i * 3 + i // 3 + j) % 9) + 1 for j in range(9)]
                    for i in range(9)]
        puzzle = [row[:] for row in solution]
        # Clear a few cells
        puzzle[0][0] = 0
        puzzle[1][1] = 0
        puzzle[2][2] = 0
        return {'puzzle': puzzle, 'solution': solution}

    def test_fills_an_empty_cell(self):
        state = self._state()
        random.seed(3)
        move = SudokuAI().generate_move(state, 'ai', 'hard')
        assert state['puzzle'][move['row']][move['col']] == 0

    def test_hard_fills_with_correct_value(self):
        random.seed(7)
        state = self._state()
        correct_count = 0
        for _ in range(50):
            move = SudokuAI().generate_move(state, 'ai', 'hard')
            if move['value'] == state['solution'][move['row']][move['col']]:
                correct_count += 1
        assert correct_count >= 42

    def test_move_shape_matches_sudoku_apply_move(self):
        """SudokuGame.apply_move reads row/col/value."""
        random.seed(1)
        move = SudokuAI().generate_move(self._state(), 'ai', 'hard')
        assert set(move.keys()) == {'row', 'col', 'value'}
        assert 0 <= move['row'] < 9
        assert 0 <= move['col'] < 9
        assert 1 <= move['value'] <= 9

    def test_raises_when_puzzle_complete(self):
        solution = [[1] * 9 for _ in range(9)]
        state = {'puzzle': [row[:] for row in solution], 'solution': solution}
        with pytest.raises(ValueError, match='No empty cells'):
            SudokuAI().generate_move(state, 'ai', 'medium')

    def test_raises_when_state_missing(self):
        with pytest.raises(ValueError, match='missing puzzle'):
            SudokuAI().generate_move({}, 'ai', 'medium')


# ─── Registry & dispatcher ───────────────────────────────────────────

class TestRegistry:
    def test_get_game_ai_resolves_direct_engines(self):
        assert isinstance(get_game_ai('trivia'), TriviaAI)
        assert isinstance(get_game_ai('opentdb_trivia'), OpenTDBTriviaAI)
        assert isinstance(get_game_ai('word_scramble'), WordScrambleAI)
        assert isinstance(get_game_ai('word_search'), WordSearchAI)
        assert isinstance(get_game_ai('sudoku'), SudokuAI)

    def test_get_game_ai_returns_none_for_board_and_phaser(self):
        assert get_game_ai('boardgame') is None
        assert get_game_ai('phaser') is None

    def test_get_game_ai_returns_none_for_unknown(self):
        assert get_game_ai('nonexistent_engine') is None

    def test_is_client_authoritative_for_boardgame(self):
        assert is_client_authoritative('boardgame') is True

    def test_is_client_authoritative_for_phaser(self):
        assert is_client_authoritative('phaser') is True

    def test_is_client_authoritative_false_for_trivia(self):
        assert is_client_authoritative('trivia') is False


class TestDispatcher:
    def test_rejects_boardgame_with_client_side_hint(self):
        with pytest.raises(ValueError, match='client-authoritative'):
            generate_ai_move({}, 'boardgame', 'ai', 'medium')

    def test_rejects_phaser_with_client_side_hint(self):
        with pytest.raises(ValueError, match='client-authoritative'):
            generate_ai_move({}, 'phaser', 'ai', 'medium')

    def test_raises_for_unknown_engine(self):
        with pytest.raises(ValueError, match='No server-side AI'):
            generate_ai_move({}, 'nonexistent_engine', 'ai', 'medium')

    def test_generates_trivia_move_end_to_end(self):
        random.seed(1)
        state = {
            'current_question_idx': 0,
            'questions': [
                {'q': 'Q?', 'a': 'yes', 'options': ['yes', 'no']},
            ],
        }
        move = generate_ai_move(state, 'trivia', 'ai', 'hard')
        assert 'answer' in move

    def test_default_difficulty_is_medium(self):
        random.seed(1)
        state = {
            'current_question_idx': 0,
            'questions': [
                {'q': 'Q?', 'a': 'x', 'options': ['x', 'y', 'z']},
            ],
        }
        # Default difficulty kwarg should not raise and should produce
        # a valid move shape.
        move = generate_ai_move(state, 'trivia', 'ai')
        assert 'answer' in move

    def test_client_authoritative_set_contains_expected(self):
        assert 'boardgame' in CLIENT_AUTHORITATIVE_ENGINES
        assert 'phaser' in CLIENT_AUTHORITATIVE_ENGINES
        assert 'trivia' not in CLIENT_AUTHORITATIVE_ENGINES


# ─── Purity: AI must never mutate game_state ────────────────────────

class TestPurity:
    """The whole architecture rests on the guarantee that game_ai
    never mutates the session's game_state — the endpoint is
    read-only, and the returned move is submitted via the normal
    submit_move path which does its own mutation. These tests assert
    that guarantee so a future regression is caught immediately.
    """

    def test_trivia_does_not_mutate_state(self):
        state = {
            'current_question_idx': 0,
            'questions': [
                {'q': 'Q?', 'a': 'yes', 'options': ['yes', 'no']},
            ],
        }
        before = copy.deepcopy(state)
        random.seed(1)
        MultipleChoiceTriviaAI().generate_move(state, 'ai', 'medium')
        assert state == before

    def test_word_scramble_does_not_mutate_state(self):
        state = {
            'current_round_idx': 0,
            'rounds': [{'word': 'planet', 'scrambled': 'telnap',
                        'solved_by': None}],
        }
        before = copy.deepcopy(state)
        random.seed(1)
        WordScrambleAI().generate_move(state, 'ai', 'medium')
        assert state == before

    def test_word_search_does_not_mutate_state(self):
        state = {
            'words_to_find': ['tiger', 'eagle', 'shark'],
            'found_words': {'tiger': 'alice'},
        }
        before = copy.deepcopy(state)
        random.seed(1)
        WordSearchAI().generate_move(state, 'ai', 'medium')
        assert state == before

    def test_sudoku_does_not_mutate_state(self):
        solution = [[((i * 3 + i // 3 + j) % 9) + 1 for j in range(9)]
                    for i in range(9)]
        puzzle = [row[:] for row in solution]
        puzzle[0][0] = 0
        state = {'puzzle': puzzle, 'solution': solution}
        before = copy.deepcopy(state)
        random.seed(1)
        SudokuAI().generate_move(state, 'ai', 'medium')
        assert state == before

    def test_dispatcher_does_not_mutate_state(self):
        state = {
            'current_question_idx': 0,
            'questions': [
                {'q': 'Q?', 'a': 'x', 'options': ['x', 'y']},
            ],
        }
        before = copy.deepcopy(state)
        random.seed(1)
        generate_ai_move(state, 'trivia', 'ai', 'medium')
        assert state == before


# ─── Registry aliases: trivia/opentdb_trivia/quick_match share handler ──

class TestRegistryAliases:
    def test_all_trivia_aliases_share_one_instance(self):
        """DRY: trivia, opentdb_trivia, and quick_match resolve to the
        same handler instance, so there's no per-engine drift."""
        trivia = get_game_ai('trivia')
        opentdb = get_game_ai('opentdb_trivia')
        quick = get_game_ai('quick_match')
        assert trivia is opentdb
        assert trivia is quick
        assert trivia is not None


# ─── Catalog-ID resolution path ─────────────────────────────────────

class TestCatalogResolution:
    """The catalog lookup branch in get_game_ai and
    is_client_authoritative must resolve catalog IDs to the right
    engine. Tested via monkeypatch to avoid coupling to the live
    catalog contents.
    """

    def test_catalog_id_resolves_to_registered_ai(self, monkeypatch):
        fake_catalog = {'fake-trivia-id': 'trivia'}
        monkeypatch.setattr(
            'integrations.social.game_catalog.get_engine_for_catalog_entry',
            lambda game_id: fake_catalog.get(game_id),
        )
        assert get_game_ai('fake-trivia-id') is get_game_ai('trivia')

    def test_catalog_id_for_boardgame_is_client_authoritative(self, monkeypatch):
        fake_catalog = {'fake-tictactoe-id': 'boardgame'}
        monkeypatch.setattr(
            'integrations.social.game_catalog.get_engine_for_catalog_entry',
            lambda game_id: fake_catalog.get(game_id),
        )
        assert is_client_authoritative('fake-tictactoe-id') is True
        assert get_game_ai('fake-tictactoe-id') is None

    def test_unknown_catalog_id_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            'integrations.social.game_catalog.get_engine_for_catalog_entry',
            lambda game_id: None,
        )
        assert get_game_ai('unknown-catalog-id') is None
        assert is_client_authoritative('unknown-catalog-id') is False
