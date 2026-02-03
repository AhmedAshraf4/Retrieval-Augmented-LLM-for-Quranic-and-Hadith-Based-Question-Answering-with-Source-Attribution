docker --version
docker compose version
docker compose up -d
docker exec -it ollama ollama pull nomic-embed-text # 2k
docker exec -it ollama ollama pull embeddinggemma:300m # 2k
docker exec -it ollama ollama pull snowflake-arctic-embed2:568m # 8k
docker exec -it ollama ollama pull dengcao/Qwen3-Embedding-0.6B:Q8_0 # 32k
docker exec -it ollama ollama pull dengcao/Qwen3-Embedding-8B:Q5_K_M # 40k
docker exec -it ollama ollama pull llama3.2

curl http://localhost:11434/api/embeddings \
  -d '{
    "model": "nomic-embed-text",
    "prompt": "hello world"
  }'
docker exec -it postgres psql -U rag -d ragdb
