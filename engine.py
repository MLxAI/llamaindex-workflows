from llama_index.core import (
    VectorStoreIndex
)
import qdrant_client
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter


def get_qdrant_vector_store(collection_name: str):
    client=qdrant_client.AsyncQdrantClient(url="10.0.78.175", port=6333)
    vector_store = QdrantVectorStore(
        aclient=client, collection_name=collection_name, enable_hybrid=True
    )
    return vector_store
    
def create_engine(doc_id: str):
    vector_store=get_qdrant_vector_store()
    vector_storage_index = VectorStoreIndex.from_vector_store(
            vector_store
        )
    filters_ = MetadataFilters(
            filters=[ExactMatchFilter(key="document_metadata_id", value=doc_id)]
        )
    engine = vector_storage_index.as_query_engine(
                similarity_top_k=100,
                filters=filters_,
                sparse_top_k=20,
                vector_store_query_mode="hybrid",
            )
    query_engine_tools=[QueryEngineTool.from_defaults(
                engine,
                name=f"workflows_tool",
                description=f"",
            )]
    return query_engine_tools