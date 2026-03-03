from typing import Any

from task.tools.base import BaseTool
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class DeleteMemoryTool(BaseTool):
    """
    Tool for deleting all long-term memories about the user.

    This permanently removes all stored memories from the system.
    Use with caution - this action cannot be undone.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        # provide self-descriptive name
        return "long_term_memory_delete_tool"

    @property
    def description(self) -> str:
        # provide tool description that will help LLM to understand when to use this tools and cover 'tricky'
        #  moments (not more 1024 chars)
        return ("Tool for deleting all long-term memories about the user. "
                "This permanently removes all stored memories from the system. ")

    @property
    def parameters(self) -> dict[str, Any]:
        # provide tool parameters JSON Schema with empty properties
        return {
            "type": "object",
            "properties": {
                # no parameters needed for this tool
             }
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str:

        # 1. Call `memory_store` `delete_all_memories` (we will implement logic in `memory_store` later
        result = await self.memory_store.delete_all_memories(tool_call_params.api_key)
        
        # 2. Add result to stage
        tool_call_params.stage.append_content(result)
        
        # 3. Return result
        return result