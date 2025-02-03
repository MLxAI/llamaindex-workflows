"""
/*--------------------------------------------------------------
 * Name: schema_validator.py
 * Version: v1.0
 * Author: Rohan
 * Organization: MLX Organization
 * --------------------------------------------------------------
 * Copyright (c) 2023 MLX Organization
 * All rights reserved.
 * --------------------------------------------------------------
"""


"""Schemas for the chat app."""
from pydantic import BaseModel, validator
from typing import List, Optional, Dict
from data_util import get_current_time


class ChatResponse(BaseModel):
    """Chat response schema."""

    sender: str
    message: str
    analysed_page: str=None
    message_created: str = None
    image_links: List[str] = []
    query_id: Optional[str] = None
    prediction_count: Optional[str] = None
    type: Optional[str] = None
    top_sources: Optional[Dict] = None
    related_question: Optional[str] = None
    message_id: Optional[str] = None
    is_last_event: bool = False
    event_id: Optional[str] = None
    metadata_map: dict | None = None
    chat_type: str = None
    code_interpreter_output: bool = False
    page_source_mappings: dict = {}

    @validator("message_created", pre=True, always=True)
    def set_message_created(cls, v):
        return v or get_current_time()

    @validator("sender")
    def sender_must_be_bot_or_you(cls, v):
        if v not in ["user", "assistant", "assistant_error"]:
            raise ValueError("sender must be user or you")
        return v

    @validator("type")
    def validate_message_type(cls, v):
        if v not in [
            "start",
            "stream",
            "end",
            "error",
            "page_analysis_result",
            "info",
            "upgrade",
            "source_doc",
            "summary_response",
            "intermediate_response",
            "page_number",
            "language",
            "response_1",
            "response_2",
            "image_resp",
            "text_resp",
            "limit_exceed",
            "related_questions",
            "assistant",
            "claude_enterprise",
            "events",
            "intermediate_events",
            "response"
        ]:
            raise ValueError("type must be start, stream or end")
        return v