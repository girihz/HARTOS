"""Tests for draft-first speculative dispatch (Qwen3.5-0.8B first-responder).

Covers:
  - ModelRegistry.get_draft_model() selects the DRAFT tier only
  - get_fast_model() and get_expert_model() now exclude DRAFT tier
  - dispatch_draft_first parses valid JSON envelope from draft output
  - dispatch_draft_first handles prose-wrapped / fenced / malformed envelopes
  - delegate routing: 'none' → no expert; 'local' → fast expert; 'hive' → expert fallback to fast
  - Recursive loop prevention: inner _dispatch_to_model sends speculative=False, draft_first=False
  - WorldModelBridge.record_interaction is called with the draft model_id tag
  - Graceful fallback when no draft model is registered
  - Circuit-breaker / constitutional filter short-circuit paths
"""
import os
import sys
import json
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def fresh_registry():
    """Fresh isolated ModelRegistry per test — no singleton contamination."""
    from integrations.agent_engine.model_registry import ModelRegistry, ModelBackend, ModelTier
    reg = ModelRegistry()
    reg.register(ModelBackend(
        model_id='qwen3.5-0.8b-draft',
        display_name='Qwen3.5 0.8B (Draft)',
        tier=ModelTier.DRAFT,
        config_list_entry={'model': 'Qwen3.5-0.8B-Instruct', 'api_key': 'dummy',
                           'base_url': 'http://localhost:8080/v1', 'price': [0, 0]},
        avg_latency_ms=300.0, accuracy_score=0.45, cost_per_1k_tokens=0.0,
        is_local=True,
    ))
    reg.register(ModelBackend(
        model_id='qwen3.5-4b-local',
        display_name='Qwen3.5 4B (Fast)',
        tier=ModelTier.FAST,
        config_list_entry={'model': 'Qwen3.5-4B', 'api_key': 'dummy',
                           'base_url': 'http://localhost:8080/v1', 'price': [0, 0]},
        avg_latency_ms=700.0, accuracy_score=0.60, cost_per_1k_tokens=0.0,
        is_local=True,
    ))
    reg.register(ModelBackend(
        model_id='gpt-4-expert',
        display_name='GPT-4 (Expert)',
        tier=ModelTier.EXPERT,
        config_list_entry={'model': 'gpt-4', 'api_key': 'dummy',
                           'base_url': 'https://api.openai.com/v1', 'price': [0.03, 0.06]},
        avg_latency_ms=3000.0, accuracy_score=0.90, cost_per_1k_tokens=60.0,
    ))
    return reg


@pytest.fixture
def dispatcher(fresh_registry):
    """SpeculativeDispatcher bound to the fresh registry."""
    from integrations.agent_engine.speculative_dispatcher import SpeculativeDispatcher
    return SpeculativeDispatcher(model_registry=fresh_registry)


# ═══════════════════════════════════════════════════════════════════════════
# Registry tier selection
# ═══════════════════════════════════════════════════════════════════════════


class TestRegistryTierSelection:

    def test_get_draft_model_returns_draft_tier(self, fresh_registry):
        draft = fresh_registry.get_draft_model()
        assert draft is not None
        assert draft.model_id == 'qwen3.5-0.8b-draft'

    def test_get_draft_model_none_when_absent(self):
        from integrations.agent_engine.model_registry import ModelRegistry, ModelBackend, ModelTier
        reg = ModelRegistry()
        reg.register(ModelBackend(
            model_id='only-fast',
            display_name='Only Fast',
            tier=ModelTier.FAST,
            config_list_entry={'model': 'x', 'api_key': 'y'},
            avg_latency_ms=500.0, accuracy_score=0.6,
        ))
        assert reg.get_draft_model() is None

    def test_get_fast_model_excludes_draft(self, fresh_registry):
        """DRAFT tier must NEVER be returned by get_fast_model — even if it
        has the lowest latency, because the draft can't produce final
        answers for complex tasks."""
        fast = fresh_registry.get_fast_model()
        assert fast is not None
        assert fast.model_id == 'qwen3.5-4b-local'
        assert fast.tier.value != 'draft'

    def test_get_expert_model_excludes_draft(self, fresh_registry):
        expert = fresh_registry.get_expert_model()
        assert expert is not None
        assert expert.tier.value != 'draft'
        assert expert.model_id == 'gpt-4-expert'

    def test_draft_has_lower_latency_than_fast(self, fresh_registry):
        draft = fresh_registry.get_draft_model()
        fast = fresh_registry.get_fast_model()
        assert draft.avg_latency_ms < fast.avg_latency_ms


# ═══════════════════════════════════════════════════════════════════════════
# JSON envelope parser
# ═══════════════════════════════════════════════════════════════════════════


class TestDraftEnvelopeParser:

    def test_clean_json(self, dispatcher):
        raw = '{"reply": "hi there", "delegate": "none", "confidence": 0.9, "reason": "greeting"}'
        parsed = dispatcher._parse_draft_envelope(raw)
        assert parsed['reply'] == 'hi there'
        assert parsed['delegate'] == 'none'
        assert parsed['confidence'] == 0.9

    def test_markdown_fenced(self, dispatcher):
        raw = '```json\n{"reply": "standby", "delegate": "local", "confidence": 0.5}\n```'
        parsed = dispatcher._parse_draft_envelope(raw)
        assert parsed['reply'] == 'standby'
        assert parsed['delegate'] == 'local'

    def test_prose_wrapped_json(self, dispatcher):
        raw = 'Here is my answer: {"reply": "ok", "delegate": "hive", "confidence": 0.3} hope it helps!'
        parsed = dispatcher._parse_draft_envelope(raw)
        assert parsed['delegate'] == 'hive'

    def test_trailing_comma_tolerated(self, dispatcher):
        raw = '{"reply": "x", "delegate": "none", "confidence": 1.0,}'
        parsed = dispatcher._parse_draft_envelope(raw)
        assert parsed.get('reply') == 'x'

    def test_completely_malformed_returns_empty(self, dispatcher):
        assert dispatcher._parse_draft_envelope('not json at all') == {}

    def test_empty_input_returns_empty(self, dispatcher):
        assert dispatcher._parse_draft_envelope('') == {}
        assert dispatcher._parse_draft_envelope(None) == {}


# ═══════════════════════════════════════════════════════════════════════════
# dispatch_draft_first — delegate routing
# ═══════════════════════════════════════════════════════════════════════════


def _mock_guardrails(monkeypatch):
    """Bypass the real guardrail network so tests stay hermetic."""
    import security.hive_guardrails as hg
    monkeypatch.setattr(hg.HiveCircuitBreaker, 'is_halted', lambda: False)
    monkeypatch.setattr(
        hg.ConstitutionalFilter, 'check_prompt',
        staticmethod(lambda p: (True, ''))
    )


class TestDispatchDraftFirstDelegation:

    def test_delegate_none_returns_draft_reply_no_expert(
            self, dispatcher, monkeypatch):
        _mock_guardrails(monkeypatch)
        draft_raw = '{"reply": "Hello!", "delegate": "none", "confidence": 0.95}'
        with patch.object(dispatcher, '_dispatch_to_model',
                          return_value=draft_raw) as mock_dispatch, \
             patch.object(dispatcher, '_record_interaction_safely') as mock_record, \
             patch.object(dispatcher._expert_pool, 'submit') as mock_submit:
            result = dispatcher.dispatch_draft_first(
                'hi', user_id='u1', prompt_id='p1')

        assert result['response'] == 'Hello!'
        assert result['delegate'] == 'none'
        assert result['draft_model'] == 'qwen3.5-0.8b-draft'
        assert result['expert_pending'] is False
        mock_dispatch.assert_called_once()
        mock_submit.assert_not_called()
        # Draft interaction was recorded for continual learning
        mock_record.assert_called_once()
        call_kwargs = mock_record.call_args.kwargs
        assert call_kwargs['model_id'] == 'qwen3.5-0.8b-draft'
        assert call_kwargs['response'] == 'Hello!'

    def test_delegate_local_fires_background_expert(
            self, dispatcher, monkeypatch):
        _mock_guardrails(monkeypatch)
        draft_raw = '{"reply": "Working on it...", "delegate": "local", "confidence": 0.4}'
        with patch.object(dispatcher, '_dispatch_to_model',
                          return_value=draft_raw), \
             patch.object(dispatcher, '_record_interaction_safely'), \
             patch.object(dispatcher, '_check_and_reserve_budget',
                          return_value=True), \
             patch.object(dispatcher._expert_pool, 'submit') as mock_submit:
            result = dispatcher.dispatch_draft_first(
                'write a function to reverse a list', user_id='u1', prompt_id='p1')

        assert result['delegate'] == 'local'
        assert result['response'] == 'Working on it...'
        assert result['expert_pending'] is True
        # Expert task was scheduled
        mock_submit.assert_called_once()
        # The scheduled model is the FAST tier (4B), not the draft or expert
        submit_args = mock_submit.call_args.args
        scheduled_model = submit_args[4]  # expert_model is the 5th positional arg
        assert scheduled_model.tier.value == 'fast'

    def test_delegate_hive_prefers_expert_tier(
            self, dispatcher, monkeypatch):
        _mock_guardrails(monkeypatch)
        draft_raw = '{"reply": "Let me check the hive...", "delegate": "hive", "confidence": 0.2}'
        with patch.object(dispatcher, '_dispatch_to_model',
                          return_value=draft_raw), \
             patch.object(dispatcher, '_record_interaction_safely'), \
             patch.object(dispatcher, '_check_and_reserve_budget',
                          return_value=True), \
             patch.object(dispatcher._expert_pool, 'submit') as mock_submit:
            result = dispatcher.dispatch_draft_first(
                'prove the Riemann hypothesis', user_id='u1', prompt_id='p1')

        assert result['delegate'] == 'hive'
        assert result['expert_pending'] is True
        scheduled_model = mock_submit.call_args.args[4]
        # Hive delegate picks the EXPERT tier (highest accuracy)
        assert scheduled_model.tier.value == 'expert'

    def test_delegate_hive_falls_back_to_fast_when_no_expert(self, monkeypatch):
        """If only a draft + fast are registered and delegate='hive',
        the dispatcher falls back to the fast tier instead of returning
        no expert at all."""
        from integrations.agent_engine.model_registry import ModelRegistry, ModelBackend, ModelTier
        from integrations.agent_engine.speculative_dispatcher import SpeculativeDispatcher
        reg = ModelRegistry()
        reg.register(ModelBackend(
            model_id='draft-only', display_name='D', tier=ModelTier.DRAFT,
            config_list_entry={'model': 'd', 'api_key': 'x'},
            avg_latency_ms=200.0, accuracy_score=0.4, is_local=True,
        ))
        reg.register(ModelBackend(
            model_id='fast-only', display_name='F', tier=ModelTier.FAST,
            config_list_entry={'model': 'f', 'api_key': 'x'},
            avg_latency_ms=700.0, accuracy_score=0.6, is_local=True,
        ))
        d = SpeculativeDispatcher(model_registry=reg)
        _mock_guardrails(monkeypatch)
        draft_raw = '{"reply": "trying", "delegate": "hive", "confidence": 0.2}'
        with patch.object(d, '_dispatch_to_model', return_value=draft_raw), \
             patch.object(d, '_record_interaction_safely'), \
             patch.object(d, '_check_and_reserve_budget', return_value=True), \
             patch.object(d._expert_pool, 'submit') as mock_submit:
            result = d.dispatch_draft_first('hard question', user_id='u1', prompt_id='p1')

        assert result['delegate'] == 'hive'
        assert result['expert_pending'] is True
        assert mock_submit.call_args.args[4].model_id == 'fast-only'

    def test_malformed_json_defaults_to_local_delegate(
            self, dispatcher, monkeypatch):
        """Tiny models sometimes emit prose instead of JSON — we should
        NOT silently treat that as 'none' (which would hide the failure).
        Default to 'local' so a bigger model confirms the answer."""
        _mock_guardrails(monkeypatch)
        with patch.object(dispatcher, '_dispatch_to_model',
                          return_value='I think the answer is 42.'), \
             patch.object(dispatcher, '_record_interaction_safely'), \
             patch.object(dispatcher, '_check_and_reserve_budget',
                          return_value=True), \
             patch.object(dispatcher._expert_pool, 'submit') as mock_submit:
            result = dispatcher.dispatch_draft_first(
                'whatever', user_id='u1', prompt_id='p1')

        assert result['delegate'] == 'local'
        assert result['response'].startswith('I think')
        mock_submit.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# Recursive-loop prevention
# ═══════════════════════════════════════════════════════════════════════════


class TestLoopPrevention:

    def test_inner_dispatch_forbids_speculative(
            self, dispatcher, monkeypatch):
        """_dispatch_to_model's recursive /chat call MUST include
        speculative=False and draft_first=False in the request body, or
        draft-first would re-enter itself via the inner call."""
        _mock_guardrails(monkeypatch)
        captured_body = {}
        def fake_post(url, **kwargs):
            captured_body.update(kwargs.get('json', {}))
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {'response': '{}'}
            return resp

        with patch('requests.post', side_effect=fake_post):
            dispatcher._dispatch_to_model(
                dispatcher._registry.get_draft_model(),
                'some prompt', user_id='u1', prompt_id='p1',
                goal_type='general', goal_id=None,
            )

        assert captured_body.get('speculative') is False
        assert captured_body.get('draft_first') is False


# ═══════════════════════════════════════════════════════════════════════════
# Graceful degradation paths
# ═══════════════════════════════════════════════════════════════════════════


class TestGracefulDegradation:

    def test_no_draft_model_returns_error_marker(self, monkeypatch):
        """When no DRAFT model is registered, dispatch_draft_first returns
        an empty-response result with error='no_draft_model'. The chat
        route checks for empty response and falls through to the normal
        path."""
        from integrations.agent_engine.model_registry import ModelRegistry
        from integrations.agent_engine.speculative_dispatcher import SpeculativeDispatcher
        empty_reg = ModelRegistry()
        d = SpeculativeDispatcher(model_registry=empty_reg)
        _mock_guardrails(monkeypatch)
        result = d.dispatch_draft_first('hi', user_id='u1', prompt_id='p1')
        assert result['response'] == ''
        assert result['error'] == 'no_draft_model'
        assert result['delegate'] == 'none'

    def test_circuit_breaker_halted_returns_empty(
            self, dispatcher, monkeypatch):
        import security.hive_guardrails as hg
        monkeypatch.setattr(hg.HiveCircuitBreaker, 'is_halted', lambda: True)
        result = dispatcher.dispatch_draft_first(
            'hi', user_id='u1', prompt_id='p1')
        assert result['response'] == ''
        assert result['error'] == 'Hive is halted'

    def test_guardrail_blocked_prompt_returns_empty(
            self, dispatcher, monkeypatch):
        import security.hive_guardrails as hg
        monkeypatch.setattr(hg.HiveCircuitBreaker, 'is_halted', lambda: False)
        monkeypatch.setattr(
            hg.ConstitutionalFilter, 'check_prompt',
            staticmethod(lambda p: (False, 'blocked by test'))
        )
        result = dispatcher.dispatch_draft_first(
            'anything', user_id='u1', prompt_id='p1')
        assert result['response'] == ''
        assert result['error'] == 'blocked by test'

    def test_record_interaction_swallows_exceptions(self, dispatcher):
        """_record_interaction_safely must never raise — otherwise a
        broken WorldModelBridge would break the chat path."""
        with patch('integrations.agent_engine.world_model_bridge.get_world_model_bridge',
                   side_effect=RuntimeError('bridge unavailable')):
            # Should not raise
            dispatcher._record_interaction_safely(
                user_id='u1', prompt_id='p1', prompt='x',
                response='y', model_id='m', latency_ms=10,
                node_id=None, goal_id=None,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Energy + latency tracking still flows through the registry
# ═══════════════════════════════════════════════════════════════════════════


class TestTelemetryHooks:

    def test_latency_recorded_on_registry(self, dispatcher, monkeypatch):
        _mock_guardrails(monkeypatch)
        with patch.object(dispatcher, '_dispatch_to_model',
                          return_value='{"reply": "ok", "delegate": "none"}'), \
             patch.object(dispatcher, '_record_interaction_safely'), \
             patch.object(dispatcher._registry, 'record_latency') as mock_latency, \
             patch.object(dispatcher._registry, 'record_energy') as mock_energy:
            dispatcher.dispatch_draft_first('hi', user_id='u1', prompt_id='p1')

        mock_latency.assert_called_once()
        mock_energy.assert_called_once()
        assert mock_latency.call_args.args[0] == 'qwen3.5-0.8b-draft'
