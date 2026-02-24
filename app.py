import spacy
import pdfplumber
import uvicorn
from fastapi import FastAPI, File, UploadFile
from io import BytesIO

# --- 1. CONFIGURATION: WORDS TO IGNORE ---
# This list tells the AI: "If you find these words, do NOT count them as skills."
IGNORE_WORDS = {
    "John", "Doe", "Jane", "Smith", "Email", "Phone", "Address", 
    "Senior", "Junior", "Software", "Engineer", "Developer", "Manager",
    "Experience", "Education", "Summary", "Skills", "University", "College",
    "Tech", "Corp", "Languages", "Databases", "Cloud", "Science"
}

# --- 2. LOAD THE AI MODEL ---
try:
    nlp = spacy.load("output/model-best")
    print("✅ Custom AI Model loaded successfully.")
except OSError:
    print("⚠️ Model not found. Loading blank English model for testing.")
    nlp = spacy.blank("en")

app = FastAPI()

def extract_text_from_pdf(file_bytes):
    """Helper function to pull text from a PDF file."""
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

@app.post("/analyze_resume")
async def analyze_resume(file: UploadFile = File(...)):
    # --- 3. READ THE FILE ---
    file_content = await file.read()
    
    text = ""
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_content)
    else:
        # Fallback for txt files
        text = file_content.decode("utf-8")

    # --- 4. RUN THE AI ---
    doc = nlp(text)
    
    # --- 5. FILTER RESULTS (THE "BOUNCER" LOGIC) ---
    found_skills = set()
    
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            # Clean the text (remove extra spaces)
            clean_skill = ent.text.strip()
            
            # CHECK 1: Is it in our Ignore List?
            if clean_skill in IGNORE_WORDS:
                continue 
            
            # CHECK 2: Does it look like an email or website?
            if "@" in clean_skill or ".com" in clean_skill:
                continue

            # If it passes checks, add it!
            found_skills.add(clean_skill)

    # --- 6. RETURN RESULTS ---
    return {
        "filename": file.filename,
        "found_skills": list(found_skills),
        "total_skills": len(found_skills),
        "raw_text_preview": text[:200] + "..." 
    }

# Run the server if executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)   