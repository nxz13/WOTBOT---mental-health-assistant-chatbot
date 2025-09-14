import os
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from gpt4all import GPT4All
import re
import textwrap
import time

#text wrapping for chatbot output
def format_reply(reply, width=80):
    return "\n".join(textwrap.wrap(reply, width=width))

transcripts_dir = "transcripts"
chroma_dir = "chroma_db"
collection_name = "company_courses"
embedder_model = "all-MiniLM-L6-v2"
chunk_size = 500
chunk_overlap = 100


model_path = r"C:\Users\msnas\AppData\Local\nomic.ai\GPT4All"
model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"

top_k = 7
min_score = 0.2 #minimum score to recommend a course


def load_transcripts():
   base_dir=transcripts_dir
   splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
   all_chunks = []
   metadatas = []
   for course in os.listdir(base_dir):
       course_path = os.path.join(base_dir, course)
       if os.path.isdir(course_path):
            for file in os.listdir(course_path):
                if file.endswith(".txt"):
                    with open(os.path.join(course_path, file), "r", encoding="utf-8") as f:
                        raw = f.read()
                        chunks = splitter.split_text(raw)
                        all_chunks.extend(chunks)
                        metadatas.extend([{"course": course}] * len(chunks))
   return all_chunks, metadatas

def build_chroma(chunks, metadatas, rebuild=False):
    client = chromadb.PersistentClient(path=chroma_dir, settings=Settings(anonymized_telemetry=False))
    if rebuild:
        for c in client.list_collections():
            if c.name == collection_name:
                client.delete_collection(collection_name)
                break
        collection = client.create_collection(name=collection_name)
        embedder = SentenceTransformer(embedder_model)
        for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
            vec = embedder.encode(chunk).tolist()
            collection.add(documents=[chunk], embeddings=[vec], ids=[str(i)], metadatas=[meta])
        return collection, embedder
    else:
        collection = client.get_or_create_collection(name=collection_name)
        embedder = SentenceTransformer(embedder_model) 
        return collection, embedder
    
def retrieve_context_course(user_input, collection, embedder, min_score=min_score):
    if not user_input.strip():
        return "", None
    
    query_vec = embedder.encode(user_input).tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=top_k, include=["documents", "metadatas", "distances"])
    if not results or not results.get("documents") or not results["documents"][0]:
        return "", None

    docs, metas, dists = results["documents"][0], results["metadatas"][0], results["distances"][0]

    words = set(re.findall(r'\w+', user_input.lower()))
    course_scores = Counter()
    snippet_by_course = {}

    for doc, meta, dist in zip(docs, metas, dists):
        if dist is None:
            continue
        similarity = 1 - dist
        snippet_words = set(re.findall(r'\w+', doc.lower()))
        overlap = len(words & snippet_words)

        score = similarity + 0.1 * overlap
        course_name = meta.get("course")

        course_scores[course_name] += score
        snippet_by_course[course_name] = snippet_by_course.get(course_name, "") + " " + doc
    
    if not course_scores:
        return "", None

    best_course, best_score = course_scores.most_common(1)[0]
    if best_score < min_score:
        return "", None

    context_text = snippet_by_course[best_course].strip()
    return context_text, best_course

def is_crisis(text):
    crisis_keywords = ["suicide","kill","die", "self-harm","self harm", "kill myself", "end my life", "hurt myself", "i want to die", "suicidal", "selfexit", "self exit"]
    t = text.lower()
    return any(k in t for k in crisis_keywords)

def is_mentalhealth_intent(text):
    keywords = [
        "anxiety","depression","stress","mental health","sad","worried",
        "overwhelmed","panic","fatigue","insomnia","burnout","adhd","ocd",
        "bipolar","autism","psychosis","therapy","counseling", "depressed"
    ]
    t = text.lower()
    return any(k in t for k in keywords)

def simple_sentiment(text):
    t = text.lower()
    neg = ["sad","down","low","bad","tired","stressed","overwhelmed","anxious","burnout","restless", "confused"]
    pos = ["good","great","okay","fine","happy","better","alright"]
    neg_score = sum(1 for k in neg if k in t)
    pos_score = sum(1 for k in pos if k in t)
    if neg_score > pos_score: return "negative"
    if pos_score > neg_score: return "positive"
    return "neutral"

def list_course_names(base_dir=transcripts_dir):
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

#Prompt Engineering
def build_system_prompt(course_list):
    courses = ", ".join(sorted(course_list)) if course_list else "None"
    return (
        "You are WOTBOT, an empathetic mental health assistant for Company A whose name is confidential.\n"
        "Provide relevant mental health advice to the user's query. Reference transcript context when available. Keep replies short and warm.\n"
        "Recommend courses only from the Company A course list when a strong match exists.\n"
        f"Available courses: {courses}\n"
        "Do not hallucinate or provide courses outside this list."
    )

def generate_reply(model, message, context, course, tone):
    if is_crisis(message):
        return (
            "I'm very concerned for your safety. Please contact local emergency services or Samaritans (UK: 116 123). You’re not alone."
        )

    tone_hint = {
        "negative":"Be gentle and validating.",
        "neutral":"Be warm and neutral.",
        "positive":"Be encouraging but concise."
        }[tone]

    if is_mentalhealth_intent(message) and context and course:
        prompt = (
            f"User query: {message}\n\nRelevant transcript snippet (course: {course}):\n{context}\n\n"
            "Respond accurately, briefly, kindly, and include course suggestion if appropriate."
        )
    elif is_mentalhealth_intent(message):
        prompt = f"User query: {message}\n\n{tone_hint}\nProvide brief empathetic advice. No course recommendation."
    else:
        prompt = f"User query: {message}\nProvide a short friendly reply. No courses."

    return model.generate(prompt, max_tokens=220, temp=0.6).strip()


def main():
    
    collection, embedder = build_chroma(None, None, rebuild=False)

    if collection.count()==0:
        chunks, metas = load_transcripts()
        collection, embedder = build_chroma(chunks, metas, rebuild=True)
    
    print("Loading GPT4All model…")
    model = GPT4All(model_name=model_name, model_path=model_path)
    print("Model loaded.\n")


    courses = list_course_names()
    system_prompt = build_system_prompt(courses)
    
    print("WOTBOT ready! Type 'bye' or 'exit' to quit.\n")
    
    with model.chat_session(system_prompt=system_prompt):
        while True:
            user = input("You: ").strip()
            if user.lower() in {"exit","quit","bye","reset"}:
                print("WOTBOT : Take care! ")
                
                break
            context, course = retrieve_context_course(user, collection, embedder)
            tone = simple_sentiment(user)
            start_time = time.time()
            reply = generate_reply(model, user, context, course, tone)
            end_time = time.time()
            response_time = end_time - start_time
            
            print(f"\nWOTBOT 🤖: {format_reply(reply)}\n")
            print(f"Response time: {response_time:.2f}s")
            

if __name__ == "__main__":
    main()
