import os
os.environ['OMP_NUM_THREADS'] = '1'

import json
from datetime import datetime, UTC, timedelta
import numpy as np
import faiss
from aidial_client import AsyncDial
from sentence_transformers import SentenceTransformer

from task.tools.memory._models import Memory, MemoryData, MemoryCollection


class LongTermMemoryStore:
    """
    Manages long-term memory storage for users.

    Storage format: Single JSON file per user in DIAL bucket
    - File: {user_id}/long-memories.json
    - Caching: In-memory cache with conversation_id as key
    - Deduplication: O(n log n) using FAISS batch search
    """

    DEDUP_INTERVAL_HOURS = 24

    def __init__(self, endpoint: str):

        # 1. Set endpoint
        self.endpoint = endpoint
        
        # 2. Create SentenceTransformer as model, model name is `all-MiniLM-L6-v2`
        self.model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")
        
        # 3. Create cache, doct of str and MemoryCollection (it is imitation of cache, normally such cache should be set aside)
        self.cache:dict[str, MemoryCollection] = {}
        
        # 4. Make `faiss.omp_set_num_threads(1)` (without this set up you won't be able to work in debug mode in `_deduplicate_fast` method
        faiss.omp_set_num_threads(1)

    async def _get_memory_file_path(self, dial_client: AsyncDial) -> str:
        """Get the path to the memory file in DIAL bucket."""

        # 1. Get DIAL app home path
        bucket_with_app_home = await dial_client.my_appdata_home()
        
        # 2. Return string with path in such format: `files/{bucket_with_app_home}/__long-memories/data.json`
        #    The memories will persist in appdata for this agent in `__long-memories` folder and `data.json` file
        #    (You will be able to check it also in Chat UI in attachments)
        return f'files/{bucket_with_app_home}/__long-memories/data.json'

    async def _load_memories(self, api_key: str) -> MemoryCollection:
        
        # 1. Create AsyncDial client (api_version is 2025-01-01-preview)
        dial_client = AsyncDial(base_url=self.endpoint, api_key=api_key, api_version='2025-01-01-preview')
        # 2. Get memory file path
        memory_file_path = await self._get_memory_file_path(dial_client)
        
        # 3. Check cache: cache is dict of str and MemoryCollection, for the key we will use `memory file path` to make
        #    it simple. Such key will be unique for user and will allow to access memories across different
        #    conversations and only user can access them. In case if cache is present return its MemoryCollection.
        # ---
        # Below is logic when cache is not present:
        # 4. Open try-except block:
        #   - in try:
        #       - download file content
        #       - in response get content and decode it with 'utf-8'
        #       - load content with `json`
        #       - create MemoryCollection (it is pydentic model, use `model_validate` method)
        #   - in except:
        #       - create MemoryCollection (it will have empty memories, set up time for updated_at, more detailed take
        #         a look at MemoryCollection pydentic model and it Fields)
        # 5. Return created MemoryCollection
        if memory_file_path in self.cache:
            return self.cache[memory_file_path]
        else:
            try:
                response = await dial_client.files.download(memory_file_path)
                content = response.get_content().decode('utf-8')
                data = json.loads(content)
                memory_collection = MemoryCollection.model_validate(data)
            except Exception as e:
                memory_collection = MemoryCollection(memories=[], updated_at=datetime.now(UTC))
        
        self.cache[memory_file_path] = memory_collection

        return memory_collection

    async def _save_memories(self, api_key: str, memories: MemoryCollection):
        """Save memories to DIAL bucket and update cache."""
        # 1. Create AsyncDial client
        dial_client = AsyncDial(base_url=self.endpoint, api_key=api_key, api_version='2025-01-01-preview')

        # 2. Get memory file path
        memory_file_path = await self._get_memory_file_path(dial_client)

        # 3. Update `updated_at` of memories (now)
        memories.updated_at = datetime.now(UTC)
        
        # 4. Converts memories to json string (it's pydentic model and it have model dump json method for this). Don't
        #    make any indentations because it will make file 'bigger'. Here is the point that we store all the memories
        #    in one file and 'one memory' with its embeddings takes ~6-8Kb, we expect that there are won't be more that
        #    1000 memories but anyway for 1000 memories it will be ~6-8Mb, so, we need to make at least these small
        #    efforts to make it smaller 😉
        content = memories.model_dump_json()
        bytes_content = content.encode('utf-8')

        await dial_client.files.upload(file=bytes_content, url=memory_file_path)
        
        # 5. Put to cache (kind reminder the key is memory file path)
        self.cache[memory_file_path] = memories

        print(f"Saved memories to {memory_file_path}")

    async def add_memory(self, api_key: str, content: str, importance: float, category: str, topics: list[str]) -> str:
        """Add a new memory to storage."""

        # 1. Load memories
        collection = await self._load_memories(api_key)

        # 2. Make encodings for content with embedding model.
        #    Hint: provide content as list, and after encoding get first result (encode wil return list) and convertit `tolist`
        embedding = self.model.encode([content])[0].tolist()
        
        # 3. Create Memory
        #    - for id use `int(datetime.now(UTC).timestamp())` it will provide time now as int, it will be super enough
        #      to avoid collisions. Also, we won't use id but we added it because maybe in future you will make enhanced
        #      version of long-term memory and after that it will be additional 'headache' to add such ids 😬
        memory = Memory(
            data=MemoryData(
                id=int(datetime.now(UTC).timestamp()),
                content=content,
                importance=importance,
                category=category,
                topics=topics
            ),
            embedding=embedding
        )        

        # 4. Add to memories created memory
        collection.memories.append(memory)

        # 5. Save memories (it is PUT request bzw, -> https://dialx.ai/dial_api#tag/Files/operation/uploadFile)
        await self._save_memories(api_key, collection)
        
        # 6. Return information that content has been successfully stored
        return f"Successfully stored memory: {content}"

    async def search_memories(self, api_key: str, query: str, top_k: int = 5) -> list[MemoryData]:
        """
        Search memories using semantic similarity.

        Returns:
            List of MemoryData objects (without embeddings)
        """
        # 1. Load memories
        collection = await self._load_memories(api_key)
        # 2. If they are empty return empty array
        if not collection.memories:
            return []
        
        # ---
        # 3. Check if they needs_deduplication, if yes then deduplicate_and_save (need to implements both of these methods)
        if self._needs_deduplication(collection):
            collection = await self._deduplicate_and_save(api_key, collection)

        # 4. Make vector search (embeddings are part of memory)😈
        embeddings = np.array([m.embedding for m in collection.memories]).astype('float32')
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
        index.add(normalized_embeddings)

        # 5. Return `top_k` MemoryData based on vector search        
        query_embedding = self.model.encode([query])[0].astype('float32')
        query_norm = np.linalg.norm(query_embedding, keepdims=True)
        normalized_query_embedding = query_embedding / query_norm
        _, indices = index.search(normalized_query_embedding, min(top_k, len(collection.memories)))

        result = [collection.memories[i].data for i in indices[0]]    

        return result

    def _needs_deduplication(self, collection: MemoryCollection) -> bool:
        """Check if deduplication is needed (>24 hours since last deduplication)."""

        # The criteria for deduplication (collection length > 10 and >24 hours since last deduplication) or
        # (collection length > 10 last deduplication is None)
        if len(collection.memories) > 10:
            if collection.last_deduplicated_at is None:
                return True
            elif datetime.now(UTC) - collection.last_deduplicated_at > timedelta(hours=self.DEDUP_INTERVAL_HOURS):
                return True
        return False

    async def _deduplicate_and_save(self, api_key: str, collection: MemoryCollection) -> MemoryCollection:
        """
        Deduplicate memories synchronously and save the result.
        Returns the updated collection.
        """
        # 1. Make fast deduplication (need to implement)
        deduplicated_memories = self._deduplicate_fast(collection.memories)
        # 2. Update last_deduplicated_at as now
        collection.last_deduplicated_at = datetime.now(UTC)
        # 3. Save deduplicated memories
        collection.memories = deduplicated_memories
        await self._save_memories(api_key, collection)
        # 4. Return deduplicated collection
        return collection

    def _deduplicate_fast(self, memories: list[Memory]) -> list[Memory]:
        """
        Fast deduplication using FAISS batch search with cosine similarity.

        Strategy:
        - Find k nearest neighbors for each memory using cosine similarity
        - Mark duplicates based on similarity threshold (cosine similarity > 0.75)
        - Keep memory with higher importance
        """

        # This is the hard part 🔥🔥🔥
        # You need to deduplicate memories, duplicates are the memories that have 75% similarity.
        # Among duplicates remember about `importance`, most important have more priorities to survive
        # It must be fast, it is possible to do for O(n log n), probably you can find faster way (share with community if do 😉)
        # Return deduplicated memories
        if len(memories) < 2:
            return memories
        
        embeddings = np.array([m.embedding for m in memories]).astype('float32')
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
        index.add(normalized_embeddings)

        k = min(5, len(memories))  # Number of neighbors to check
        similarities, indices = index.search(normalized_embeddings, k)  # Get top k neighbors for each memory

        dublicates_to_remove = set()
        for i in range(len(memories)):

            for j in range(1, k):  # Start from 1 to skip self-comparison
                neighbor_idx = indices[i][j]

                if neighbor_idx in dublicates_to_remove:
                    continue

                if similarities[i][j] > 0.75:  # Similarity threshold
                    if memories[i].data.importance >= memories[neighbor_idx].data.importance:
                        dublicates_to_remove.add(neighbor_idx)
                    else:
                        dublicates_to_remove.add(i)
                        break  # No need to check other neighbors if current memory is marked as duplicate

        deduplicated = [m for i, m in enumerate(memories) if i not in dublicates_to_remove]
        return deduplicated

    async def delete_all_memories(self, api_key: str, ) -> str:
        """
        Delete all memories for the user.

        Removes the memory file from DIAL bucket and clears the cache
        for the current conversation.
        """
        # 1. Create AsyncDial client
        dial_client = AsyncDial(base_url=self.endpoint, api_key=api_key)

        # 2. Get memory file path
        file_path = await self._get_memory_file_path(dial_client)

        # 3. Delete file
        try:
            await dial_client.delete_file(file_path)
        except Exception as e:
            print(f"Warning: Could not delete memory file: {e}")

        if file_path in self._cache:
            del self._cache[file_path]
            print(f"Cleared memory cache: {file_path}")

        # 4. Return info about successful memory deletion
        return f"All memories deleted successfully from {file_path}"