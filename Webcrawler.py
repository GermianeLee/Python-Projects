import sys
import os
import re
import requests
from bs4 import BeautifulSoup
import sqlite3
import math
import time
from urllib.parse import urlparse
from collections import defaultdict
from porterstemmer import PorterStemmer  # Import PorterStemmer

# Define stop words (from IndexerUnit4.py)
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will',
    'with', 'i', 'you', 'we', 'they', 'this', 'but', 'or', 'not', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own',
    'same', 'so', 'than', 'too', 'very'
}

# Global counters
documents = 0
tokens = 0
terms = 0
stop_words_count = 0
crawled_urls = 0  # Track successfully crawled URLs
failed_urls = 0   # Track failed URLs

# Database dictionary for in-memory processing
database = {}
term_block_size = 50000  # Approximate term limit per block
current_block_terms = 0  # Counter for terms in the current block

# Regular expressions
chars = re.compile(r'\W+')  # Split on non-word characters
pattid = re.compile(r'(\d{3})/(\d{3})/(\d{3})')  # Extract ID from path
punct_start = re.compile(r'^\W')  # Terms starting with punctuation
number_term = re.compile(r'^\d+$')  # Terms that are numbers

# Term class to store term information
class Term:
    def __init__(self):
        self.termid = 0
        self.termfreq = 0
        self.docs = 0
        self.docids = {}

# Initialize Porter Stemmer
stemmer = PorterStemmer()

# Split line into tokens
def splitchars(line):
    return chars.split(line)

# Write block to disk
def write_block_to_disk(cur, con):
    global database, current_block_terms
    for term, term_obj in database.items():
        cur.execute("INSERT INTO TermDictionary VALUES (?, ?)", (term, term_obj.termid))
        for docid, tf in term_obj.docids.items():
            df = term_obj.docs
            idf = math.log(documents / df) if df > 0 else 0
            tfidf = tf * idf
            cur.execute("INSERT INTO Posting VALUES (?, ?, ?, ?, ?)",
                        (term_obj.termid, docid, tfidf, df, tf))
    con.commit()
    database.clear()
    current_block_terms = 0

# Process tokens from a web page
def parsetoken(line, cur, con):
    global documents, tokens, terms, stop_words_count, current_block_terms
    line = line.replace('\t', ' ').strip()
    l = splitchars(line)
    for elmt in l:
        elmt = elmt.replace('\n', '').lower().strip()
        if not elmt:
            continue
        tokens += 1
        if elmt in STOP_WORDS:
            stop_words_count += 1
            continue
        if (punct_start.match(elmt) or
                number_term.match(elmt) or
                len(elmt) <= 2):
            continue
        stemmed = stemmer.stem(elmt, 0, len(elmt) - 1)
        if not stemmed:
            continue
        if stemmed not in database:
            terms += 1
            current_block_terms += 1
            database[stemmed] = Term()
            database[stemmed].termid = terms
            database[stemmed].docids = {}
            database[stemmed].docs = 0
        if documents not in database[stemmed].docids:
            database[stemmed].docs += 1
            database[stemmed].docids[documents] = 0
        database[stemmed].docids[documents] += 1
        if current_block_terms >= term_block_size:
            write_block_to_disk(cur, con)
    return l

# Search engine functions (from IndexerSearchEngineUnit5.2.py)
def process_query(query, cur):
    """Process query into stemmed, filtered terms."""
    terms = splitchars(query.lower())
    valid_terms = []
    raw_terms = []
    for term in terms:
        if (term not in STOP_WORDS and
                len(term) > 2 and
                not number_term.match(term) and
                not punct_start.match(term)):
            stemmed = stemmer.stem(term, 0, len(term) - 1)
            if stemmed:
                valid_terms.append(stemmed)
                raw_terms.append(term)
    return valid_terms, raw_terms

def calculate_query_tf_idf(terms, cur):
    """Calculate tf-idf weights for query terms."""
    term_freq = defaultdict(int)
    query_tf_idf = {}
    total_docs = cur.execute("SELECT COUNT(*) FROM DocumentDictionary").fetchone()[0]
    for term in terms:
        term_freq[term] += 1
    for term, freq in term_freq.items():
        cur.execute("SELECT TermId FROM TermDictionary WHERE Term = ?", (term,))
        term_id = cur.fetchone()
        if term_id:
            cur.execute("SELECT docfreq FROM Posting WHERE TermId = ? LIMIT 1", (term_id[0],))
            df = cur.fetchone()
            if df and df[0] > 0:
                idf = math.log(total_docs / df[0])
                query_tf_idf[term] = (1 + math.log10(freq)) * idf
    return query_tf_idf

def get_document_vector(doc_id, terms, cur):
    """Retrieve tf-idf weights for document terms."""
    doc_vector = {}
    for term in terms:
        cur.execute("SELECT TermId FROM TermDictionary WHERE Term = ?", (term,))
        term_id = cur.fetchone()
        if term_id:
            cur.execute("SELECT tfidf FROM Posting WHERE TermId = ? AND DocId = ?", (term_id[0], doc_id))
            tfidf = cur.fetchone()
            if tfidf:
                doc_vector[term] = tfidf[0]
    return doc_vector

def cosine_similarity(query_tf_idf, doc_id, terms, cur):
    """Calculate cosine similarity between query and document (Simpson algorithm)."""
    doc_vector = get_document_vector(doc_id, terms, cur)
    dot_product = 0
    query_norm = 0
    doc_norm = 0
    for term, q_weight in query_tf_idf.items():
        d_weight = doc_vector.get(term, 0)
        dot_product += q_weight * d_weight
        query_norm += q_weight ** 2
        doc_norm += d_weight ** 2
    query_norm = math.sqrt(query_norm)
    doc_norm = math.sqrt(doc_norm)
    if query_norm == 0 or doc_norm == 0:
        return 0
    return dot_product / (query_norm * doc_norm)

def search(query, cur, con):
    """Process query and return top 20 documents by cosine similarity."""
    valid_terms, raw_terms = process_query(query, cur)
    if not valid_terms:
        return [], 0, []

    query_tf_idf = calculate_query_tf_idf(valid_terms, cur)
    if not query_tf_idf:
        return [], 0, []

    candidate_docs = None
    for term in valid_terms:
        cur.execute("SELECT TermId FROM TermDictionary WHERE Term = ?", (term,))
        term_id = cur.fetchone()
        if term_id:
            cur.execute("SELECT DocId FROM Posting WHERE TermId = ?", (term_id[0],))
            doc_ids = set(row[0] for row in cur.fetchall())
            if candidate_docs is None:
                candidate_docs = doc_ids
            else:
                candidate_docs &= doc_ids

    if not candidate_docs:
        return [], 0, []

    similarities = []
    term_matches = []
    for doc_id in candidate_docs:
        sim_score = cosine_similarity(query_tf_idf, doc_id, valid_terms, cur)
        if sim_score > 0:
            cur.execute("SELECT DocumentName FROM DocumentDictionary WHERE DocId = ?", (doc_id,))
            doc_name = cur.fetchone()[0]
            doc_terms = []
            for stemmed, raw in zip(valid_terms, raw_terms):
                cur.execute("SELECT TermId FROM TermDictionary WHERE Term = ?", (stemmed,))
                term_id = cur.fetchone()
                if term_id:
                    cur.execute("SELECT DocId FROM Posting WHERE TermId = ? AND DocId = ?", (term_id[0], doc_id))
                    if cur.fetchone():
                        doc_terms.append(raw)
            similarities.append((doc_id, doc_name, sim_score))
            term_matches.append((doc_name, doc_terms))

    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:20], len(similarities), term_matches

def search_main(db_path):
    """Run the search engine."""
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        print("\nStarting search engine...")
        while True:
            query = input("Enter query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            results, total_candidates, term_matches = search(query, cur, con)
            print(f"\nSearch Results (Simpson Algorithm, {total_candidates} candidates retrieved):")
            for i, (doc_id, doc_name, score) in enumerate(results, 1):
                print(f"{i}. Document: {doc_name}, Cosine Similarity: {score:.4f}")
                for match_doc_name, terms in term_matches:
                    if match_doc_name == doc_name and terms:
                        for term in terms:
                            print(f'Found "{term}" in document: {doc_name}')
            if not results:
                print("No matching documents found.")
        con.close()
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
        sys.exit(1)

def main():
    global documents, crawled_urls, failed_urls
    # Prompt for starting URL
    starting_url = input("Enter URL to crawl (must be in the form http://www.domain.com): ")
    db_path = "webcrawler.db"

    # Initialize database
    try:
        con = sqlite3.connect(db_path)
        con.isolation_level = None
    except sqlite3.OperationalError as e:
        print(f"Failed to connect to database at {db_path}: {e}")
        sys.exit(1)
    cur = con.cursor()

    # Create database tables
    cur.execute("DROP TABLE IF EXISTS DocumentDictionary")
    cur.execute("DROP INDEX IF EXISTS idxDocumentDictionary")
    cur.execute("CREATE TABLE DocumentDictionary (DocumentName TEXT, DocId INT)")
    cur.execute("CREATE INDEX idxDocumentDictionary ON DocumentDictionary (DocId)")

    cur.execute("DROP TABLE IF EXISTS TermDictionary")
    cur.execute("DROP INDEX IF EXISTS idxTermDictionary")
    cur.execute("CREATE TABLE TermDictionary (Term TEXT, TermId INT)")
    cur.execute("CREATE INDEX idxTermDictionary ON TermDictionary (TermId)")

    cur.execute("DROP TABLE IF EXISTS Posting")
    cur.execute("DROP INDEX IF EXISTS idxPosting1")
    cur.execute("DROP INDEX IF EXISTS idxPosting2")
    cur.execute("CREATE TABLE Posting (TermId INT, DocId INT, tfidf REAL, docfreq INT, termfreq INT)")
    cur.execute("CREATE INDEX idxPosting1 ON Posting (TermId)")
    cur.execute("CREATE INDEX idxPosting2 ON Posting (DocId)")

    # Initialize crawling variables
    crawled = []
    tocrawl = [starting_url]
    links_queue = 0
    crawlcomplete = True

    print(f"Start Time: {time.strftime('%H:%M')}")

    # Crawl loop (DFS)
    while crawlcomplete:
        try:
            crawling = tocrawl.pop()
        except IndexError:
            crawlcomplete = False
            continue

        if crawling.endswith(('.pdf', '.png', '.jpg', '.gif', '.asp')):
            crawled.append(crawling)
            continue

        print(f"Queue size: {len(tocrawl)}, Crawling: {crawling}")

        try:
            response = requests.get(crawling, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text() for p in paragraphs if p.get_text())
            # Process text and update index
            parsetoken(text, cur, con)
            documents += 1
            cur.execute("INSERT INTO DocumentDictionary VALUES (?, ?)", (crawling, documents))
            crawled_urls += 1  # Increment successful crawl
        except requests.RequestException as e:
            print(f"Failed to fetch {crawling}: {e}")
            failed_urls += 1  # Increment failed crawl
            crawled.append(crawling)
            continue

        if links_queue < 500:
            parsed_url = urlparse(crawling)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/'):
                    href = f"{base_url}{href}"
                elif href.startswith('#'):
                    href = f"{crawling}{href}"
                elif not href.startswith('http'):
                    href = f"{base_url}/{href}"
                if href not in crawled and href not in tocrawl:
                    links_queue += 1
                    if links_queue <= 500:
                        tocrawl.append(href)
        crawled.append(crawling)

    if database:
        write_block_to_disk(cur, con)

    # Calculate crawl success rate
    total_attempts = crawled_urls + failed_urls
    success_rate = (crawled_urls / total_attempts * 100) if total_attempts > 0 else 0

    # Output statistics
    print(f"Indexing Complete, write to disk: {time.strftime('%H:%M')}")
    print(f"Documents {documents}")
    print(f" Unique Terms {terms}")
    print(f"Tokens {tokens}")
    print(f" Stop Words Processed {stop_words_count}")
    print(f"Crawled: {crawled_urls} URLs")
    print(f"Failed: {failed_urls} URLs")
    print(f"Crawl Success Rate: {success_rate:.0f}%")
    print(f"End Time: {time.strftime('%H:%M')}")

    con.commit()
    con.close()

    # Start search engine
    search_main(db_path)

if __name__ == '__main__':
    main()
