import spacy
import random
from spacy.training.example import Example

# 1. The Training Data (The "Textbook")
# Format: ("Sentence", {"entities": [(start_char, end_char, "LABEL")]})
TRAINING_DATA = [
    ("I have 3 years of experience in Python and Java.", {"entities": [(32, 38, "SKILL"), (43, 47, "SKILL")]}),
    ("Proficient in AWS and Docker containers.", {"entities": [(14, 17, "SKILL"), (22, 28, "SKILL")]}),
    ("Developed a REST API using Flask and PostgreSQL.", {"entities": [(27, 32, "SKILL"), (37, 47, "SKILL")]}),
    ("Knowledge of React.js for frontend development.", {"entities": [(13, 21, "SKILL")]}),
    ("Experience with CI/CD pipelines and Jenkins.", {"entities": [(36, 43, "SKILL")]}),
    ("Skilled in Machine Learning and TensorFlow.", {"entities": [(11, 27, "SKILL"), (32, 42, "SKILL")]}),
]

def train_skill_model():
    # 2. Create a blank English model
    nlp = spacy.blank("en")
    print("Created blank 'en' model")

    # 3. Create the NER (Named Entity Recognition) pipeline component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    
    # 4. Add the new label "SKILL" to the pipeline
    ner.add_label("SKILL")

    # 5. Start Training
    optimizer = nlp.begin_training()
    
    print("Training started...")
    # Iterate 20 times (epochs) over the data to learn effectively
    for i in range(20):
        random.shuffle(TRAINING_DATA)
        losses = {}
        
        # Batch the examples
        for text, annotations in TRAINING_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            
            # Update the model
            nlp.update([example], drop=0.5, losses=losses)
            
        print(f"Epoch {i+1} Loss: {losses['ner']:.4f}")

    # 6. Save the trained model to disk
    nlp.to_disk("output/model-best")
    print("\nModel saved to 'output/model-best'")

def test_model():
    # Load the model we just trained
    print("\nLoading model for testing...")
    nlp = spacy.load("output/model-best")
    
    # Test on a NEW sentence the model hasn't seen
    test_text = "I am looking for a job in Kubernetes and Golang."
    doc = nlp(test_text)
    
    print(f"\nProcessing: '{test_text}'")
    print("Detected Skills:")
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_})")

if __name__ == "__main__":
    train_skill_model()
    test_model()