import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory._models import MemoryData
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class SearchMemoryTool(BaseTool):
    """
    Tool for searching long-term memories about the user.

    Performs semantic search over stored memories to find relevant information.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store


    @property
    def name(self) -> str:
        # provide self-descriptive name
        return "long_term_memory_search_tool"

    @property
    def description(self) -> str:
        # provide tool description that will help LLM to understand when to use this tools and cover 'tricky'
        #  moments (not more 1024 chars)
        return ("Tool for searching long-term memories about the user. "
                "Performs semantic search over stored memories to find relevant information. "
                "Should be used to retrieve important context or facts about the user for personalized responses.")

    @property
    def parameters(self) -> dict[str, Any]:
        # provide tool parameters JSON Schema:
        #  - query is string, description: "The search query. Can be a question or keywords to find relevant memories", required
        #  - top_k is integer, description: "Number of most relevant memories to return.", minimum is 1, maximum is 20, default is 5
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Can be a question or keywords to find relevant memories.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant memories to return.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                }
            },
            "required": ["query"]
        }


    async def _execute(self, tool_call_params: ToolCallParams) -> str:

        # 1. Load arguments with `json`
        args = json.load(tool_call_params.tool_call.function.arguments)

        # 2. Get `query` from arguments
        query = args.get('query')

        # 3. Get `top_k` from arguments, default is 5
        top_k = args.get('top_k', 5)

        # 4. Call `memory_store` `search_memories` (we will implement logic in `memory_store` later)
        result = await self.memory_store.search_memories(api_key=tool_call_params.api_key, query=query, top_k=top_k)
        
        # 5. If results are empty then set `final_result` as "No memories found.",
        #    otherwise iterate through results and collect content, category and topics (if preset) in markdown format
        if not result:
            final_result = "No memories found."
        else:
            final_result = ""
            for memory in result:
                final_result += f"**Content:** {memory.content}\n"
                final_result += f"**Category:** {memory.category}\n"
                if memory.topics:
                    final_result += f"**Topics:** {', '.join(memory.topics)}\n"
                final_result += "\n"

        # 6. Add result to stage as markdown text
        
        tool_call_params.stage.append_content(final_result)
        # 7. Return result
        return final_result
