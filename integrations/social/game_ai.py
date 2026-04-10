"""
HevolveSocial — Server-side AI move generators for solo game play.

When a user plays solo against the computer, the AI opponent needs to
produce a move. For engines where the **server holds the answer key**
(trivia, word_scramble, word_search, sudoku), generating a move is
cheap: read the ground truth, optionally inject difficulty-scaled
mistakes, return a validated move shape.

For **board games** (the 'boardgame' engine family — tictactoe,
connect4, checkers, reversi, mancala, nim, dots_and_boxes, battleship)
the rules live client-side in the boardgame.io Game definitions
(Nunba: landing-page/src/components/Social/Games/board-games/*;
RN: components/Social/Games/board-games/*). Re-implementing those
rules here would be a DRY violation — so this module **explicitly
rejects** board-game AI requests and defers to the client's
boardgame.io MCTSBot/RandomBot. See Nunba `utils/gameAI.js`.

For **phaser** (arcade score-chasing games) there is no move to make —
the player plays against the scene itself. No server-side AI.

Difficulty model
────────────────
All AI classes accept `difficulty` ∈ {'easy', 'medium', 'hard'}. This
scales a single `error_rate` parameter that determines how often the AI
chooses a non-optimal move. Each subclass maps error_rate to its own
domain (wrong answer, wrong guess, wrong sudoku cell, etc.).

    difficulty  error_rate  interpretation
    easy        0.45        ≈ half wrong   (beatable by anyone)
    medium      0.18        ≈ 1/5 wrong    (decent challenge)
    hard        0.04        ≈ rarely wrong (near-perfect)

Usage
─────
    from integrations.social.game_ai import generate_ai_move

    move = generate_ai_move(
        game_state=session.game_state,
        game_type=session.game_type,
        ai_user_id='ai',
        difficulty='medium',
    )
    # move is the same shape expected by the corresponding game
    # type's apply_move() — the caller can feed it straight into
    # GameService.submit_move().

The caller (api_games.ai_move endpoint) is responsible for:
  - Verifying the session is active
  - Authorizing the request (host only, to prevent spam)
  - Submitting the returned move via the normal submit_move path

This module does NOT mutate session state. It is a pure move generator.

Phase 1 limitation — "host plays both sides"
─────────────────────────────────────────────
Until a dedicated AI GameParticipant + GameService.submit_move_as_ai
path lands, the /ai_move endpoint returns a move dict ready for
`POST /games/<id>/move`, BUT when that second POST runs, the move is
authenticated as the host (via g.user_id) and credited to the host's
score — not to a distinct AI opponent. For solo trivia/word-scramble/
word-search/sudoku this means "the host plays both sides": the AI's
answer is submitted under the host's identity and the host's score
rises regardless of whether the host personally answered correctly.

Phase 1 use cases where this is acceptable:
  - "Suggest-a-move" hints during a stuck human game
  - Automated smoke tests / CI for game session flows
  - Future cross-device agents previewing AI moves before display

Phase 1 use cases where this is NOT acceptable (and callers should
avoid /ai_move for now):
  - Solo ghost-opponent with a live scoreboard
  - Multiplayer fill-dropped-seat

Phase 2 work: add `ai_opponent: True` to create_session config, create
an AI GameParticipant row, add `GameService.submit_move_as_ai(ai_uid)`
that bypasses the g.user_id-must-be-participant check. The /ai_move
endpoint then moves from read-only generator to one-step submit.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger('hevolve_social')


# ─── Difficulty → error rate map ────────────────────────────────────

DIFFICULTY_ERROR_RATE = {
    'easy': 0.45,
    'medium': 0.18,
    'hard': 0.04,
}


def _error_rate(difficulty: str) -> float:
    """Resolve difficulty string to an error rate in [0, 1]."""
    return DIFFICULTY_ERROR_RATE.get(difficulty, DIFFICULTY_ERROR_RATE['medium'])


# ─── Base class ─────────────────────────────────────────────────────

class GameAI:
    """Base class for server-side AI move generators.

    Subclasses implement generate_move(game_state, ai_user_id, difficulty)
    and return a move_data dict matching the schema their corresponding
    BaseGameType.apply_move() expects.
    """

    def generate_move(self, game_state: Dict, ai_user_id: str,
                      difficulty: str = 'medium') -> Dict:
        raise NotImplementedError

    def _roll_error(self, difficulty: str) -> bool:
        """Return True iff the AI should make a mistake this turn."""
        return random.random() < _error_rate(difficulty)


# ─── Multiple-choice trivia (shared by TriviaGame + OpenTDBTriviaGame) ──

class MultipleChoiceTriviaAI(GameAI):
    """AI for both TriviaGame and OpenTDBTriviaGame.

    Both engines normalize their questions to the same internal shape
    before storing in game_state:
        {'q': str, 'a': str, 'options': [str, ...], ...}
    (see game_types.TriviaGame.CATEGORIES and
    game_types_extended.OpenTDBTriviaGame._fetch_questions lines 75-89,
    which do `html.unescape` + shuffle and emit the same 'a'/'options'
    keys as TriviaGame).

    The server stores the correct answer at
    game_state['questions'][current_question_idx]['a']. On error, the
    AI picks a different option from the question's `options` list.
    """

    def generate_move(self, game_state, ai_user_id, difficulty='medium'):
        idx = game_state.get('current_question_idx', 0)
        questions = game_state.get('questions', [])
        if idx >= len(questions):
            raise ValueError("No active question to answer")

        question = questions[idx]
        correct = question.get('a', '')
        options = question.get('options', []) or []

        if self._roll_error(difficulty) and len(options) > 1:
            wrong = [o for o in options if o != correct]
            if wrong:
                return {'answer': random.choice(wrong)}

        return {'answer': correct}


# Backwards-compatible aliases so callers can import the engine-specific
# name if they prefer — both resolve to the same implementation.
TriviaAI = MultipleChoiceTriviaAI
OpenTDBTriviaAI = MultipleChoiceTriviaAI


# ─── Word Scramble ──────────────────────────────────────────────────

class WordScrambleAI(GameAI):
    """AI for WordScrambleGame.

    The server stores the unscrambled word at
    game_state['rounds'][current_round_idx]['word']. On error, the AI
    emits a deliberately-wrong guess (permutation of the scrambled
    letters that is not the correct word).
    """

    def generate_move(self, game_state, ai_user_id, difficulty='medium'):
        idx = game_state.get('current_round_idx', 0)
        rounds = game_state.get('rounds', [])
        if idx >= len(rounds):
            raise ValueError("No active round")

        round_data = rounds[idx]
        if round_data.get('solved_by'):
            raise ValueError("Round already solved")

        correct = round_data.get('word', '')
        scrambled = round_data.get('scrambled', correct)

        move = {'word': correct, 'time_ms': self._response_time_ms(difficulty)}

        if self._roll_error(difficulty) and len(scrambled) > 1:
            letters = list(scrambled)
            for _ in range(10):
                random.shuffle(letters)
                candidate = ''.join(letters)
                if candidate.lower() != correct.lower():
                    move['word'] = candidate
                    break

        return move

    def _response_time_ms(self, difficulty: str) -> int:
        """Simulate AI 'thinking time'. Harder AIs respond faster."""
        return {
            'easy': random.randint(8000, 15000),
            'medium': random.randint(4000, 9000),
            'hard': random.randint(1500, 5000),
        }.get(difficulty, 5000)


# ─── Word Search ────────────────────────────────────────────────────

class WordSearchAI(GameAI):
    """AI for WordSearchGame.

    All hidden words are in game_state['words_to_find']. The AI picks
    an as-yet-unfound word. On error (easy mode) it occasionally picks
    an already-found word, which the engine treats as a no-op / score 0.
    """

    def generate_move(self, game_state, ai_user_id, difficulty='medium'):
        words_to_find = game_state.get('words_to_find', []) or []
        found_words = game_state.get('found_words', {}) or {}

        unfound = [w for w in words_to_find if w.lower() not in found_words]
        if not unfound:
            raise ValueError("No unfound words left")

        # On error, occasionally emit a word that has already been found
        # (engine returns 0 score — AI effectively wastes a turn).
        if self._roll_error(difficulty) and found_words:
            return {'word': random.choice(list(found_words.keys()))}

        return {'word': random.choice(unfound)}


# ─── Sudoku ─────────────────────────────────────────────────────────

class SudokuAI(GameAI):
    """AI for SudokuGame.

    The server stores the solved grid at game_state['solution']. The
    AI picks an empty cell (puzzle[r][c] == 0) and fills it with the
    correct value. On error it fills with a deliberately-wrong digit.
    """

    def generate_move(self, game_state, ai_user_id, difficulty='medium'):
        puzzle = game_state.get('puzzle', [])
        solution = game_state.get('solution', [])
        if not puzzle or not solution:
            raise ValueError("Sudoku state missing puzzle or solution")

        empty_cells: List[Tuple[int, int]] = [
            (r, c)
            for r in range(len(puzzle))
            for c in range(len(puzzle[r]))
            if puzzle[r][c] == 0
        ]
        if not empty_cells:
            raise ValueError("No empty cells remaining")

        r, c = random.choice(empty_cells)
        correct = solution[r][c]

        value = correct
        if self._roll_error(difficulty):
            wrong_values = [v for v in range(1, 10) if v != correct]
            if wrong_values:
                value = random.choice(wrong_values)

        return {'row': r, 'col': c, 'value': value}


# ─── Registry ───────────────────────────────────────────────────────

# Single shared instance for trivia-family engines — both TriviaGame
# and OpenTDBTriviaGame normalize to the same question shape, so the
# same handler works for both.
_TRIVIA_AI = MultipleChoiceTriviaAI()

_AI_REGISTRY: Dict[str, GameAI] = {
    'trivia': _TRIVIA_AI,
    'quick_match': _TRIVIA_AI,    # quick_match defaults to trivia
    'opentdb_trivia': _TRIVIA_AI,  # same normalized shape as TriviaGame
    'word_scramble': WordScrambleAI(),
    'word_search': WordSearchAI(),
    'sudoku': SudokuAI(),
    # boardgame, phaser intentionally absent — see module docstring.
    # word_chain, collab_puzzle, compute_challenge: no AI — caller
    # gets "No server-side AI registered" (distinct from
    # client-authoritative rejection).
}


# Explicit "client-authoritative" set so the API endpoint can return a
# clear error pointing callers at the client-side path instead of a
# generic 400. Matches the engines with client-side Game definitions.
CLIENT_AUTHORITATIVE_ENGINES = frozenset({
    'boardgame',  # use boardgame.io MCTSBot on client
    'phaser',     # arcade — no AI move concept
})


def _resolve_catalog_engine(game_type: str) -> Optional[str]:
    """Resolve a catalog ID to its underlying engine name, or None.

    Narrow exception catching: only ImportError is tolerated (optional
    dependency). Other exceptions (malformed catalog entries, type
    errors) propagate so mis-routing is visible rather than hidden
    behind a False fallback.
    """
    try:
        from .game_catalog import get_engine_for_catalog_entry
    except ImportError:
        return None
    return get_engine_for_catalog_entry(game_type)


def get_game_ai(game_type: str) -> Optional[GameAI]:
    """Resolve a game_type (engine name or catalog id) to an AI handler.

    Returns None when no server-side AI exists. Callers should check
    for None and either emit an error or fall back to the client-side
    path.
    """
    # Direct engine-name match
    if game_type in _AI_REGISTRY:
        return _AI_REGISTRY[game_type]

    # Catalog-id lookup — resolve to the underlying engine name
    engine = _resolve_catalog_engine(game_type)
    if engine and engine in _AI_REGISTRY:
        return _AI_REGISTRY[engine]

    return None


def is_client_authoritative(game_type: str) -> bool:
    """True if this engine's AI must run client-side (not here)."""
    if game_type in CLIENT_AUTHORITATIVE_ENGINES:
        return True
    engine = _resolve_catalog_engine(game_type)
    return bool(engine and engine in CLIENT_AUTHORITATIVE_ENGINES)


def generate_ai_move(game_state: Dict, game_type: str,
                     ai_user_id: str = 'ai',
                     difficulty: str = 'medium') -> Dict:
    """Generate an AI move for the given session state.

    Args:
        game_state: The session's current game_state dict (from
            GameSession.game_state). Not mutated.
        game_type: The session's game_type (engine name or catalog id).
        ai_user_id: The user_id string the AI will play as. Passed
            through to the game's move (handlers need it for
            turn-ownership checks).
        difficulty: 'easy' | 'medium' | 'hard'.

    Returns:
        A move_data dict ready to feed into GameService.submit_move()
        or the matching game type's apply_move().

    Raises:
        ValueError: No server-side AI exists for this game type (the
            caller should route to the client-side path), OR the game
            state is in a position where no move is possible.
    """
    if is_client_authoritative(game_type):
        raise ValueError(
            f"Game type '{game_type}' is client-authoritative — "
            "use the client-side AI (boardgame.io MCTSBot on Nunba / RN)."
        )

    ai = get_game_ai(game_type)
    if ai is None:
        raise ValueError(f"No server-side AI registered for game type '{game_type}'")

    return ai.generate_move(game_state, ai_user_id, difficulty)
