from workflows import QueryPlanningWorkflow
from engine import create_engine
from stream_handlers import ChatResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query_planner")
async def run_worflows(doc_id: str,query_id: str,query: str):
    workflow = QueryPlanningWorkflow(verbose=False, timeout=200)
    query_engine_tools=create_engine(doc_id=doc_id)
    handler = workflow.run(
        query=query,
        tools=query_engine_tools,
    )
    start_resp = ChatResponse(
                sender="assistant",
                message="",
                type="start",
                query_id=query_id,
                message_id=doc_id,
                prediction_count="",
                chat_type="planner agent",
            )
    events=""
    async for event in handler.stream_events():
        if hasattr(event, "msg"):
            events+=event.msg
            print(events)
            ChatResponse(
                message=event.msg,
                is_last_event=False,
                sender="assistant",
                type="stream",
                query_id=query_id,
                message_id=doc_id,
                chat_type="planner agent",
            ).dict()
    end_resp = ChatResponse(
                sender="assistant",
                message="",
                type="end",
                message_id=doc_id,
                is_last_event=True,
                query_id=query_id,
                prediction_count="",
                chat_type="default",
            )
    return events