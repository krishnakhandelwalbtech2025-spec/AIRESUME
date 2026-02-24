import streamlit as st
import requests

# Set page title and icon
st.set_page_config(page_title="AI Resume Analyzer", page_icon="🤖")

st.title("🤖 AI Resume Skills Extractor")
st.write("Upload a resume to see what skills the AI detects.")

# File Uploader
uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    # Button to trigger analysis
    if st.button("Analyze Resume"):
        with st.spinner("AI is reading the resume..."):
            try:
                # Prepare the file to send to the backend
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                
                # Send request to your local API (app.py must be running!)
                response = requests.post("http://127.0.0.1:8000/analyze_resume", files=files)
                
                # Handle the response
                if response.status_code == 200:
                    data = response.json()
                    skills = data.get("found_skills", [])
                    
                    st.success(f"✅ Success! Found {len(skills)} Skills")
                    
                    # Display skills nicely
                    st.write("### Detected Skills:")
                    # Create a simple visual list
                    st.write(", ".join([f"**{skill}**" for skill in skills]))
                    
                    # Show raw data for debugging
                    with st.expander("See Raw JSON Data"):
                        st.json(data)
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
                st.warning("Make sure your backend is running! (python app.py)")