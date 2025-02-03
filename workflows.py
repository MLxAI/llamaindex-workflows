from llama_index.core.workflow import (
    Workflow,
    StopEvent,
    StartEvent,
    Context,
    step,
)
from models import *
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI


class QueryPlanningWorkflow(Workflow):
    llm = OpenAI(model="gpt-4o")
    planning_prompt = PromptTemplate(
        "Think step by step. Given an initial query, as well as information about the indexes you can query, return a plan for a RAG system.\n"
        "The plan should be a list of QueryPlanItem objects, where each object contains a query.\n"
        "The result of executing an entire plan should provide a result that is a substantial answer to the initial query, "
        "or enough information to form a new query plan.\n"
        "Sources you can query: {context}\n"
        "Initial query: {query}\n"
        "Plan:"
    )
    decision_prompt = PromptTemplate(
        "Given the following information, return a final response that satisfies the original query, or return 'PLAN' if you need to continue planning.\n"
        "Original query: {query}\n"
        "Current results: {results}\n"
    )

    @step
    async def planning_step(
        self, ctx: Context, ev: StartEvent | ExecutedPlanEvent
    ) -> QueryPlanItem | StopEvent:
        if isinstance(ev, StartEvent):
            # Initially, we need to plan
            query = ev.get("query")

            tools = ev.get("tools")

            await ctx.set("tools", {t.metadata.name: t for t in tools})
            await ctx.set("original_query", query)

            context_str = "\n".join(
                [
                    f"{i+1}. {tool.metadata.name}: {tool.metadata.description}"
                    for i, tool in enumerate(tools)
                ]
            )
            await ctx.set("context", context_str)

            query_plan = await self.llm.astructured_predict(
                QueryPlan,
                self.planning_prompt,
                context=context_str,
                query=query,
            )

            ctx.write_event_to_stream(
                Event(msg=f"Planning step: {query_plan}")
            )

            num_items = len(query_plan.items)
            await ctx.set("num_items", num_items)
            for item in query_plan.items:
                ctx.send_event(item)
        else:
            # If we've already gone through planning and executing, we need to decide
            # if we should continue planning or if we can stop and return a result.
            query = await ctx.get("original_query")
            current_results_str = ev.result

            decision = await self.llm.apredict(
                self.decision_prompt,
                query=query,
                results=current_results_str,
            )

            # Simple string matching to see if we need to keep planning or if we can stop.
            if "PLAN" in decision:
                context_str = await ctx.get("context")
                query_plan = await self.llm.astructured_predict(
                    QueryPlan,
                    self.planning_prompt,
                    context=context_str,
                    query=query,
                )

                ctx.write_event_to_stream(
                    Event(msg=f"Re-Planning step: {query_plan}")
                )

                num_items = len(query_plan.items)
                await ctx.set("num_items", num_items)
                for item in query_plan.items:
                    ctx.send_event(item)
            else:
                return StopEvent(result=decision)

    @step(num_workers=4)
    async def execute_item(
        self, ctx: Context, ev: QueryPlanItem
    ) -> QueryPlanItemResult:
        tools = await ctx.get("tools")
        tool = tools[ev.name]

        ctx.write_event_to_stream(
            Event(
                msg=f"Querying tool {tool.metadata.name} with query: {ev.query}"
            )
        )

        result = await tool.acall(ev.query)

        ctx.write_event_to_stream(
            Event(msg=f"Tool {tool.metadata.name} returned: {result}")
        )

        return QueryPlanItemResult(query=ev.query, result=str(result))

    @step
    async def aggregate_results(
        self, ctx: Context, ev: QueryPlanItemResult
    ) -> ExecutedPlanEvent:
        # We need to collect the results of the query plan items to aggregate them.
        num_items = await ctx.get("num_items")
        results = ctx.collect_events(ev, [QueryPlanItemResult] * num_items)

        # collect_events returns None if not all events were found
        # return and wait for the remaining events to come in.
        if results is None:
            return

        aggregated_result = "\n------\n".join(
            [
                f"{i+1}. {result.query}: {result.result}"
                for i, result in enumerate(results)
            ]
        )
        return ExecutedPlanEvent(result=aggregated_result)