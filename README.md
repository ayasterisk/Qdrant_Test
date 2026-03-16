pip install -r requirements.txt 
docker run -p 6333:6333 qdrant/qdrant 
python scripts/ingest_qdrant.py 
streamlit run src/app.py
