# Import required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to read resume text
def read_resume(file):
    with open(file, 'r') as f:
        return f.read()

# List of resumes
resumes = ["resume1.txt", "resume2.txt"]

# Job description
job_description = "Python Machine Learning Data Science"

# Loop over resumes
for res in resumes:
    resume_text = read_resume(res)

    # --- New Scoring Logic ---
    cv = CountVectorizer()
    vectors = cv.fit_transform([job_description, resume_text]).toarray()
    similarity = cosine_similarity([vectors[0]], [vectors[1]])

    print(res, "Match Score:", similarity[0][0])

    # -----------------------------
# Resume Screening Project
# -----------------------------

# Required Libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to read resume text
def read_resume(file):
    with open(file, 'r') as f:
        return f.read()

# List of resume files
resumes = ["resume1.txt", "resume2.txt"]

# Job Description
job_description = "Python Machine Learning Data Science"

# Step 1: Calculate Scores for Each Resume
results = []

for res in resumes:
    resume_text = read_resume(res)
    
    # --- Logical / ML scoring ---
    cv = CountVectorizer()
    vectors = cv.fit_transform([job_description, resume_text]).toarray()
    similarity = cosine_similarity([vectors[0]], [vectors[1]])
    
    # Append resume and its score
    results.append((res, similarity[0][0]))

# Step 2: Sort resumes based on score (descending)
results.sort(key=lambda x: x[1], reverse=True)

# Step 3: Print Final Ranking & Best Resume
print("Final Ranking:")
for r in results:
    print(r[0], "Score:", r[1])

print("\nBest Resume:", results[0][0])