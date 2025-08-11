import sys, os, re
import sqlite3
import time

# The database is a simple dictionary
database = {}

# Regular expressions for: extracting words, extracting ID from path
chars = re.compile(r'\W+')
pattid = re.compile(r'(\d{3})/(\d{3})/(\d{3})')

# Counters
tokens = 0
documents = 0
terms = 0

# Term object for each unique term
class Term:
    def __init__(self):
        self.termid = 0
        self.termfreq = 0
        self.docs = 0
        self.docids = {}

# Split on non-word characters
def splitchars(line):
    return chars.split(line)

# Process tokens from a line
def parsetoken(line):
    global documents, tokens, terms

    # Replace tabs with spaces and strip whitespace
    line = line.replace('\t', ' ').strip()
    
    # Split the line into tokens
    tokens_list = splitchars(line)
    
    # Process each token
    for elmt in tokens_list:
        # Remove newline and convert to lowercase
        elmt = elmt.replace('\n', '').lower().strip()
        if not elmt:  # Skip empty tokens
            continue
        
        # Increment total token count
        tokens += 1
        
        # Add new term to dictionary if it doesn't exist
        if elmt not in database:
            terms += 1
            database[elmt] = Term()
            database[elmt].termid = terms
            database[elmt].docids = {}
            database[elmt].docs = 0
        
        # Add document to term's docids if not already present
        if documents not in database[elmt].docids:
            database[elmt].docs += 1
            database[elmt].docids[documents] = 0
        
        # Increment term frequency for this document
        database[elmt].docids[documents] += 1
        database[elmt].termfreq += 1
    
    return tokens_list

# Process a file
def process(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                parsetoken(line)
    except IOError:
        print(f"Error reading file {filename}")
        return False
    return True

# Recursively walk through directory
def walkdir(cur, dirname):
    global documents
    try:
        for f in os.listdir(dirname):
            path = os.path.join(dirname, f)
            if os.path.isdir(path):
                walkdir(cur, path)
            else:
                documents += 1
                cur.execute("INSERT INTO DocumentDictionary VALUES (?, ?)", (path, documents))
                process(path)
    except Exception as e:
        print(f"Error processing directory {dirname}: {e}")
    return True

# Database connection
def get_cursor():
    conn = sqlite3.connect("indexer.db")
    return conn.cursor()

# Main execution
if __name__ == '__main__':
    # Capture start time
    start_time = time.localtime()
    print(f"Start Time: {start_time.tm_hour:02d}:{start_time.tm_min:02d}")

    # Set corpus directory
    folder = r"C:\Users\GongLee73\PycharmProjects\PythonProjectCS3308\cacm"  # Updated path

    # Create SQLite database
    con = sqlite3.connect("indexer_part2.db")
    con.isolation_level = None
    cur = con.cursor()

    # Create tables
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

    # Process corpus
    walkdir(cur, folder)

    # Insert terms into TermDictionary
    for term, term_obj in sorted(database.items()):
        cur.execute("INSERT INTO TermDictionary VALUES (?, ?)", (term, term_obj.termid))

    # Commit changes
    con.commit()

    # Print TermDictionary contents
    print("The content of TermDictionary table are as follows:")
    cur.execute("SELECT * FROM TermDictionary")
    print(cur.fetchall())

    # Print DocumentDictionary contents
    print("\nHere is a listing of the rows in the DocumentDictionary table:")
    cur.execute("SELECT * FROM DocumentDictionary")
    for row in cur.fetchall():
        print(row)

    # Close database
    con.close()

    # Print statistics
    print(f"\nDocuments: {documents}")
    print(f"Terms: {terms}")
    print(f"Tokens: {tokens}")

    end_time = time.localtime()
    print(f"End Time: {end_time.tm_hour:02d}:{end_time.tm_min:02d}")