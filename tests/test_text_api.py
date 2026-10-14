"""Tests for the text chat API."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from google.genai import types
from pydantic import ValidationError

import app.main as main


def test_chat_request_rejects_blank_message():
    with pytest.raises(ValidationError):
        main.ChatRequest(message="   ")


def test_chat_request_rejects_blank_session_id():
    with pytest.raises(ValidationError):
        main.ChatRequest(message="Hello", session_id="   ")


def _install_text_api_mocks(monkeypatch, session_service, events):
    mock_agent = MagicMock()
    monkeypatch.setattr(main, "get_session_service", lambda: session_service)
    mock_create_agent = MagicMock(return_value=mock_agent)
    monkeypatch.setattr(main, "create_agent", mock_create_agent)

    run_arguments = {}

    class FakeRunner:
        def __init__(self, *, agent, app_name, session_service):
            assert agent is mock_agent

        async def run_async(self, **kwargs):
            run_arguments.update(kwargs)
            for event in events:
                yield event

    monkeypatch.setattr(main, "Runner", FakeRunner)
    return mock_create_agent, run_arguments


@pytest.mark.asyncio
async def test_text_chat_creates_session_and_returns_final_text(monkeypatch):
    session = SimpleNamespace(id="text-session-1")
    session_service = MagicMock()
    session_service.create_session = AsyncMock(return_value=session)
    session_service.get_session = AsyncMock()
    events = [
        SimpleNamespace(
            is_final_response=lambda: False,
            content=types.Content(parts=[types.Part.from_text(text="Ignore this.")]),
        ),
        SimpleNamespace(
            is_final_response=lambda: True,
            content=types.Content(parts=[types.Part.from_text(text="Hello from Trija.")]),
        ),
    ]
    mock_create_agent, run_arguments = _install_text_api_mocks(
        monkeypatch,
        session_service,
        events,
    )

    result = await main.text_chat(main.ChatRequest(message="Hello"))

    assert result.response == "Hello from Trija."
    assert result.session_id == session.id
    session_service.create_session.assert_awaited_once_with(
        app_name=main.APP_NAME,
        user_id=main.TEXT_API_USER_ID,
    )
    session_service.get_session.assert_not_awaited()
    mock_create_agent.assert_called_once_with(model=main.GEMINI_TEXT_MODEL)
    assert run_arguments["user_id"] == main.TEXT_API_USER_ID
    assert run_arguments["session_id"] == session.id
    assert run_arguments["new_message"].parts[0].text == "Hello"


@pytest.mark.asyncio
async def test_text_chat_continues_existing_session(monkeypatch):
    session = SimpleNamespace(id="existing-session")
    session_service = MagicMock()
    session_service.create_session = AsyncMock()
    session_service.get_session = AsyncMock(return_value=session)
    _install_text_api_mocks(
        monkeypatch,
        session_service,
        [
            SimpleNamespace(
                is_final_response=lambda: True,
                content=types.Content(parts=[types.Part.from_text(text="Welcome back.")]),
            )
        ],
    )

    result = await main.text_chat(
        main.ChatRequest(message="Continue", session_id=session.id)
    )

    assert result.session_id == session.id
    assert result.response == "Welcome back."
    session_service.get_session.assert_awaited_once_with(
        app_name=main.APP_NAME,
        user_id=main.TEXT_API_USER_ID,
        session_id=session.id,
    )
    session_service.create_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_chat_rejects_unknown_session(monkeypatch):
    session_service = MagicMock()
    session_service.get_session = AsyncMock(return_value=None)
    session_service.create_session = AsyncMock()
    mock_create_agent, _ = _install_text_api_mocks(
        monkeypatch,
        session_service,
        [],
    )

    with pytest.raises(HTTPException) as error:
        await main.text_chat(
            main.ChatRequest(message="Continue", session_id="unknown")
        )

    assert error.value.status_code == 404
    session_service.create_session.assert_not_awaited()
    mock_create_agent.assert_not_called()
