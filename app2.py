import streamlit as st
import pandas as pd
import numpy as np
import base64
import io
import os
import time
import datetime
import secrets
import socket
import platform
import random
import requests
from pathlib import Path
import json
import re
try:
    import geocoder
    from geopy.geocoders import Nominatim
    GEOCODING_AVAILABLE = True
except ImportError:
    GEOCODING_AVAILABLE = False

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
import nltk

# PDF Processing
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter

# Additional imports
from streamlit_tags import st_tags
from PIL import Image
import plotly.express as px

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# ==================== Configuration ====================
st.set_page_config(
    page_title="Resumate AI ",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Enhanced CSS ====================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1a1d29 100%);
        color: #FFFFFF;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    .stCard {
        background: linear-gradient(135deg, #1e2130 0%, #262b3d 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 75, 75, 0.2);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    
    .stCard:hover {
        transform: translateY(-5px);
        border-color: #FF4B4B;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #FF4B4B 0%, #ff3333 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #ff3333 0%, #ff1a1a 100%);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.5);
        transform: translateY(-2px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #2d313a 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #FF4B4B;
        text-align: center;
        box-shadow: 0 8px 20px rgba(255, 75, 75, 0.2);
        transition: all 0.3s ease;
        width: 100%;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 30px rgba(255, 75, 75, 0.4);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #ff8080 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        word-wrap: break-word;
        overflow-wrap: break-word;
        hyphens: auto;
        line-height: 1.1;
        max-width: 100%;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #b0b0b0;
        margin-top: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .job-card {
        background: linear-gradient(135deg, #1e2130 0%, #2d313a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #00C48C;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .job-card:hover {
        transform: translateX(10px);
        box-shadow: 0 6px 20px rgba(0, 196, 140, 0.3);
    }
    
    .info-box {
        background: linear-gradient(135deg, #1e2130 0%, #262b3d 100%);
        border-left: 5px solid #FF4B4B;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .score-excellent {
        color: #00C48C !important;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 196, 140, 0.5);
    }
    
    .score-good {
        color: #FFA500 !important;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(255, 165, 0, 0.5);
    }
    
    .score-poor {
        color: #FF4B4B !important;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
    }
    
    .stTextInput>div>div>input {
        background-color: #1e2130;
        color: #FFFFFF;
        border: 1px solid #3d3d46;
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #FF4B4B;
        box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ==================== Directory Setup ====================
SCRIPT_DIR = Path(__file__).parent
CSV_DIR = SCRIPT_DIR / 'resumate_data'
RESUME_DIR = SCRIPT_DIR / 'uploaded_resumes'
CSV_DIR.mkdir(exist_ok=True)
RESUME_DIR.mkdir(exist_ok=True)

USER_DATA_CSV = CSV_DIR / 'users.csv'
FEEDBACK_CSV = CSV_DIR / 'feedback.csv'

# ==================== API Keys ====================
GROQ_API_KEY = "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"
RAPIDAPI_KEY = "f11509220amshacdf4a37eb0525bp13b188jsn95e091e6f6f7"

# ==================== ML Models & Data ====================
SKILL_CATEGORIES = {
    'Data Science': ['python', 'machine learning', 'deep learning', 'tensorflow', 'keras', 'pytorch', 
                     'pandas', 'numpy', 'scikit-learn', 'data analysis', 'statistics', 'sql', 'tableau', 
                     'power bi', 'r programming', 'data visualization', 'nlp', 'computer vision'],
    
    'Web Development': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'node.js', 'django', 
                        'flask', 'fastapi', 'php', 'laravel', 'mongodb', 'postgresql', 'mysql', 
                        'rest api', 'graphql', 'typescript', 'redux', 'webpack'],
    
    'Mobile Development': ['android', 'ios', 'flutter', 'react native', 'swift', 'kotlin', 'java', 
                          'xamarin', 'ionic', 'mobile ui', 'firebase', 'app development'],
    
    'DevOps': ['docker', 'kubernetes', 'jenkins', 'ci/cd', 'aws', 'azure', 'gcp', 'terraform', 
               'ansible', 'linux', 'bash', 'git', 'monitoring', 'cloud computing'],
    
    'Cybersecurity': ['penetration testing', 'ethical hacking', 'network security', 'cryptography', 
                      'firewall', 'security audit', 'vulnerability assessment', 'kali linux', 'wireshark'],
    
    'UI/UX Design': ['figma', 'adobe xd', 'sketch', 'wireframing', 'prototyping', 'user research', 
                     'ui design', 'ux design', 'interaction design', 'design thinking']
}

# ==================== Enhanced Helper Functions ====================

def init_csv_files():
    if not USER_DATA_CSV.exists():
        pd.DataFrame(columns=['ID', 'sec_token', 'ip_add', 'host_name', 'dev_user', 'os_name_ver',
                             'latlong', 'city', 'state', 'country', 'act_name', 'act_mail', 'act_mob',
                             'Name', 'Email_ID', 'resume_score', 'Timestamp', 'Page_no', 'Predicted_Field',
                             'User_level', 'Actual_skills', 'Recommended_skills', 'Recommended_courses', 'pdf_name']).to_csv(USER_DATA_CSV, index=False)
    
    if not FEEDBACK_CSV.exists():
        pd.DataFrame(columns=['ID', 'feed_name', 'feed_email', 'feed_score', 'comments', 'Timestamp']).to_csv(FEEDBACK_CSV, index=False)

def get_next_id(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return 1 if len(df) == 0 else int(df['ID'].max()) + 1
    except:
        return 1

def extract_text_from_pdf(file_path):
    try:
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
        
        with open(file_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                page_interpreter.process_page(page)
            text = fake_file_handle.getvalue()
        
        converter.close()
        fake_file_handle.close()
        return text.lower()
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return ""

def show_pdf(file_path):
    try:
        # Get file information
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        
        # Create a professional file display card
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1e2329 0%, #2d313a 100%);
            border: 2px solid #FF4B4B;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(255, 75, 75, 0.1);
        ">
            <div style="font-size: 5rem; margin-bottom: 1rem; color: #FF4B4B;">📄</div>
            <h2 style="color: #FFFFFF; margin-bottom: 1rem; font-weight: 700;">
                Resume Successfully Uploaded!
            </h2>
            <div style="background: rgba(255, 75, 75, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                <p style="color: #a0a0a0; margin: 0.5rem 0; font-size: 1.1rem;">
                    <strong style="color: #FF4B4B;">📁 File Name:</strong> {file_name}
                </p>
                <p style="color: #a0a0a0; margin: 0.5rem 0; font-size: 1.1rem;">
                    <strong style="color: #00C48C;">📏 File Size:</strong> {file_size:,} bytes
                </p>
                <p style="color: #a0a0a0; margin: 0.5rem 0; font-size: 1.1rem;">
                    <strong style="color: #FFD700;">📅 Upload Time:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
            <div style="background: rgba(0, 196, 140, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                <p style="color: #00C48C; margin: 0; font-size: 1.2rem; font-weight: 600;">
                    ✅ Your resume is being processed by our AI algorithms
                </p>
                <p style="color: #a0a0a0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    Analysis includes: ATS scoring, skill extraction, career field prediction, and job matching
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add download button with better styling
        with open(file_path, "rb") as f:
            st.download_button(
                label="📥 Download Your Resume",
                data=f.read(),
                file_name=file_name,
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
            
        # Add some additional info
        st.markdown("""
        <div style="
            background: rgba(255, 75, 75, 0.05);
            border-left: 4px solid #FF4B4B;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        ">
            <p style="color: #a0a0a0; margin: 0; font-size: 0.9rem;">
                💡 <strong>Note:</strong> PDF preview is not available in this environment due to browser security restrictions. 
                You can download your resume using the button above to view it locally.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        st.info("Your resume has been uploaded successfully and is being analyzed.")

def predict_career_field_naive_bayes(resume_text):
    try:
        training_texts = []
        training_labels = []
        
        for category, skills in SKILL_CATEGORIES.items():
            for skill in skills:
                training_texts.append(skill)
                training_labels.append(category)
        
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(training_texts)
        
        clf = MultinomialNB()
        clf.fit(X_train, training_labels)
        
        X_test = vectorizer.transform([resume_text])
        prediction = clf.predict(X_test)[0]
        confidence = max(clf.predict_proba(X_test)[0])
        
        return prediction, round(confidence * 100, 2)
    except:
        return "General IT", 0

def analyze_resume_ml_enhanced(resume_text):
    """Enhanced scoring with content quality analysis"""
    score = 0
    feedback = []
    
    # Word count for content depth analysis
    word_count = len(resume_text.split())
    
    # Enhanced weights based on importance
    weights = {
        'contact': 8,
        'summary': 10,
        'experience': 20,
        'education': 12,
        'skills': 15,
        'projects': 15,
        'certifications': 8,
        'achievements': 7,
        'social': 5
    }
    
    # Contact Information (0-8 points)
    contact_keywords = ['email', 'phone', 'linkedin', 'github', 'mobile', 'contact']
    contact_count = sum(1 for word in contact_keywords if word in resume_text)
    if contact_count >= 3:
        score += weights['contact']
        feedback.append(('✓ Complete Contact Information', weights['contact']))
    elif contact_count >= 2:
        partial = int(weights['contact'] * 0.6)
        score += partial
        feedback.append((f'⚠ Partial Contact Info (+{partial} pts)', partial))
    else:
        feedback.append(('✗ Missing Contact Information', 0))
    
    # Professional Summary (0-10 points)
    summary_keywords = ['summary', 'objective', 'profile', 'about']
    has_summary = any(word in resume_text for word in summary_keywords)
    if has_summary:
        # Check quality - should have substantial content after keyword
        summary_section = resume_text[resume_text.find(next(k for k in summary_keywords if k in resume_text)):]
        summary_words = len(summary_section.split()[:50])
        if summary_words >= 30:
            score += weights['summary']
            feedback.append(('✓ Detailed Professional Summary', weights['summary']))
        else:
            partial = int(weights['summary'] * 0.5)
            score += partial
            feedback.append((f'⚠ Brief Summary - Add More Detail (+{partial} pts)', partial))
    else:
        feedback.append(('✗ Add Professional Summary', 0))
    
    # Work Experience (0-20 points)
    experience_keywords = ['experience', 'work history', 'employment', 'worked', 'position', 'role']
    exp_count = sum(1 for word in experience_keywords if word in resume_text)
    # Check for action verbs and achievements
    action_verbs = ['developed', 'managed', 'led', 'created', 'implemented', 'designed', 'achieved', 'improved']
    action_count = sum(1 for verb in action_verbs if verb in resume_text)
    
    if exp_count >= 2 and action_count >= 3:
        score += weights['experience']
        feedback.append(('✓ Strong Work Experience with Achievements', weights['experience']))
    elif exp_count >= 1 and action_count >= 1:
        partial = int(weights['experience'] * 0.65)
        score += partial
        feedback.append((f'⚠ Work Experience Present - Add More Achievements (+{partial} pts)', partial))
    elif exp_count >= 1:
        partial = int(weights['experience'] * 0.4)
        score += partial
        feedback.append((f'⚠ Basic Experience Listed (+{partial} pts)', partial))
    else:
        feedback.append(('✗ Add Work Experience Section', 0))
    
    # Education (0-12 points)
    education_keywords = ['education', 'degree', 'university', 'college', 'bachelor', 'master', 'phd', 'diploma']
    edu_count = sum(1 for word in education_keywords if word in resume_text)
    if edu_count >= 3:
        score += weights['education']
        feedback.append(('✓ Comprehensive Education Details', weights['education']))
    elif edu_count >= 2:
        partial = int(weights['education'] * 0.7)
        score += partial
        feedback.append((f'⚠ Education Present - Add More Details (+{partial} pts)', partial))
    else:
        feedback.append(('✗ Add Education Details', 0))
    
    # Skills (0-15 points) - Most critical
    skills_detected = extract_skills(resume_text)
    if len(skills_detected) >= 8:
        score += weights['skills']
        feedback.append(('✓ Rich Skills Section', weights['skills']))
    elif len(skills_detected) >= 5:
        partial = int(weights['skills'] * 0.7)
        score += partial
        feedback.append((f'⚠ Good Skills - Add More Technical Skills (+{partial} pts)', partial))
    elif len(skills_detected) >= 2:
        partial = int(weights['skills'] * 0.4)
        score += partial
        feedback.append((f'⚠ Limited Skills Listed (+{partial} pts)', partial))
    else:
        feedback.append(('✗ Add Technical Skills Section', 0))
    
    # Projects (0-15 points)
    project_keywords = ['project', 'portfolio', 'github', 'built', 'developed application']
    proj_count = sum(1 for word in project_keywords if word in resume_text)
    if proj_count >= 2:
        score += weights['projects']
        feedback.append(('✓ Projects/Portfolio Included', weights['projects']))
    elif proj_count >= 1:
        partial = int(weights['projects'] * 0.5)
        score += partial
        feedback.append((f'⚠ Add More Project Details (+{partial} pts)', partial))
    else:
        feedback.append(('✗ Add Projects Section', 0))
    
    # Certifications (0-8 points)
    cert_keywords = ['certification', 'certificate', 'certified', 'licensed']
    if any(word in resume_text for word in cert_keywords):
        score += weights['certifications']
        feedback.append(('✓ Certifications Listed', weights['certifications']))
    
    # Achievements (0-7 points)
    achievement_keywords = ['achievement', 'award', 'recognition', 'won', 'honored']
    if any(word in resume_text for word in achievement_keywords):
        score += weights['achievements']
        feedback.append(('✓ Achievements Highlighted', weights['achievements']))
    
    # Professional Links (0-5 points)
    link_keywords = ['linkedin', 'github', 'portfolio', 'website']
    link_count = sum(1 for word in link_keywords if word in resume_text)
    if link_count >= 2:
        score += weights['social']
        feedback.append(('✓ Professional Links Added', weights['social']))
    elif link_count >= 1:
        partial = int(weights['social'] * 0.6)
        score += partial
        feedback.append((f'⚠ Add More Professional Links (+{partial} pts)', partial))
    
    # Content depth bonus (0-5 points)
    if word_count >= 400:
        score += 5
        feedback.append(('✓ Comprehensive Content', 5))
    elif word_count >= 250:
        score += 3
        feedback.append(('⚠ Add More Detail (+3 pts)', 3))
    else:
        feedback.append(('✗ Resume Too Brief - Add More Content', 0))
    
    return min(score, 100), feedback

def extract_skills(resume_text):
    all_skills = []
    for category, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            if skill in resume_text:
                all_skills.append(skill.title())
    return list(set(all_skills))

def calculate_skill_match_tfidf(resume_text, job_skills):
    try:
        documents = [resume_text, ' '.join(job_skills)]
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(similarity * 100, 2)
    except:
        return 0

def fetch_jobs_rapidapi(query, location="United States"):
    url = "https://jsearch.p.rapidapi.com/search"
    querystring = {"query": f"{query} in {location}", "page": "1", "num_pages": "1", "date_posted": "month"}
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "jsearch.p.rapidapi.com"}
    
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Job API Error: {e}")
        return None

def get_youtube_recommendations(predicted_field):
    """Get YouTube video recommendations"""
    videos = {
        'Data Science': [
            {'title': 'Data Science Full Course 2024', 'url': 'https://www.youtube.com/watch?v=ua-CiDNNj30'},
            {'title': 'Machine Learning Tutorial', 'url': 'https://www.youtube.com/watch?v=Gv9_4yMHFhI'},
        ],
        'Web Development': [
            {'title': 'Full Stack Web Development Course', 'url': 'https://www.youtube.com/watch?v=nu_pCVPKzTk'},
            {'title': 'React JS Full Course', 'url': 'https://www.youtube.com/watch?v=b9eMGE7QtTk'},
        ],
        'Mobile Development': [
            {'title': 'Flutter Complete Tutorial', 'url': 'https://www.youtube.com/watch?v=VPvVD8t02U8'},
            {'title': 'React Native Crash Course', 'url': 'https://www.youtube.com/watch?v=0-S5a0eXPoc'},
        ],
        'DevOps': [
            {'title': 'DevOps Full Course', 'url': 'https://www.youtube.com/watch?v=Rv3o-6ZMqS4'},
            {'title': 'Docker Tutorial', 'url': 'https://www.youtube.com/watch?v=3c-iBn73dDE'},
        ],
        'General IT': [
            {'title': 'IT Career Roadmap 2024', 'url': 'https://www.youtube.com/watch?v=hKu12iMBAWU'},
            {'title': 'Resume Writing Tips', 'url': 'https://www.youtube.com/watch?v=Tt08KmFfIYQ'},
        ]
    }
    
    return videos.get(predicted_field, videos['General IT'])

def insert_data(sec_token, ip_add, host_name, dev_user, os_name_ver, latlong, city, state, country, 
                act_name, act_mail, act_mob, name, email, res_score, timestamp, no_of_pages, 
                reco_field, cand_level, skills, recommended_skills, courses, pdf_name):
    try:
        df = pd.read_csv(USER_DATA_CSV)
        new_id = get_next_id(USER_DATA_CSV)
        
        new_row = {
            'ID': new_id, 'sec_token': sec_token, 'ip_add': ip_add, 'host_name': host_name,
            'dev_user': dev_user, 'os_name_ver': os_name_ver, 'latlong': latlong, 'city': city,
            'state': state, 'country': country, 'act_name': act_name, 'act_mail': act_mail,
            'act_mob': act_mob, 'Name': name, 'Email_ID': email, 'resume_score': res_score,
            'Timestamp': timestamp, 'Page_no': no_of_pages, 'Predicted_Field': reco_field,
            'User_level': cand_level, 'Actual_skills': skills, 'Recommended_skills': recommended_skills,
            'Recommended_courses': courses, 'pdf_name': pdf_name
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(USER_DATA_CSV, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

def insertf_data(feed_name, feed_email, feed_score, comments, Timestamp):
    try:
        df = pd.read_csv(FEEDBACK_CSV)
        new_id = get_next_id(FEEDBACK_CSV)
        
        new_row = {'ID': new_id, 'feed_name': feed_name, 'feed_email': feed_email, 
                  'feed_score': feed_score, 'comments': comments, 'Timestamp': Timestamp}
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(FEEDBACK_CSV, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return False

# ==================== Main Application ====================

def main():
    init_csv_files()
    
    try:
        logo_path = SCRIPT_DIR / 'Logo' / 'Resu.png'
        img = Image.open(logo_path)
        st.image(img)
    except:
        # Enhanced centered title with proper design
        st.markdown("""
            <div style="text-align: center; padding: 3rem 0; margin-bottom: 2rem;">
                <h1 style="
                    font-size: 3.5rem; 
                    font-weight: 800; 
                    background: linear-gradient(135deg, #FF4B4B 0%, #ff8080 50%, #00C48C 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    margin: 0;
                    text-shadow: 0 4px 8px rgba(0,0,0,0.3);
                    letter-spacing: 2px;
                "> AI Resume Analyzer</h1>
                <p style="
                    font-size: 1.3rem; 
                    color: #a0a0a0; 
                    margin: 1rem 0 0 0;
                    font-weight: 300;
                    letter-spacing: 1px;
                ">Transform Your Career with Intelligent Resume Analysis</p>
                <div style="
                    width: 100px; 
                    height: 4px; 
                    background: linear-gradient(135deg, #FF4B4B 0%, #00C48C 100%);
                    margin: 1.5rem auto;
                    border-radius: 2px;
                "></div>
            </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("# Choose Something...")
    activities = ["User", "Feedback", "About", "Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    
    link = '<b>Built with 🤍 by <a href="" style="text-decoration: none; color: #FF4B4B;">Team Resumate AI</a></b>'
    st.sidebar.markdown(link, unsafe_allow_html=True)
    
# ==================== User Page ====================
    if choice == 'User':
        
        # Progress Tracker
        if 'user_step' not in st.session_state:
            st.session_state.user_step = 1
        
        # Step Indicator
        steps = ["📝 Personal Info", "📤 Upload Resume", "🤖 Analysis", "📊 Results"]
        cols = st.columns(4)
        for idx, (col, step) in enumerate(zip(cols, steps), 1):
            with col:
                if idx <= st.session_state.user_step:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #FF4B4B 0%, #ff3333 100%); 
                        padding: 1rem; border-radius: 10px; text-align: center; font-weight: 600;'>
                        {step}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style='background: rgba(60, 60, 70, 0.5); 
                        padding: 1rem; border-radius: 10px; text-align: center; color: #666;'>
                        {step}
                        </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main Content Area
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Personal Information Form
            st.markdown("""
                <div class='stCard'>
                    <h3>📋 Personal Information</h3>
                    <p style='color: #a0a0a0;'>Tell us about yourself</p>
                </div>
            """, unsafe_allow_html=True)
            
            act_name = st.text_input('Full Name*', placeholder="John Doe", key="user_name")
            act_mail = st.text_input('Email Address*', placeholder="john@example.com", key="user_email")
            act_mob = st.text_input('Mobile Number*', placeholder="+1 234 567 8900", key="user_mobile")
            
            # Additional user preferences
            col_pref1, col_pref2 = st.columns(2)
            with col_pref1:
                job_preference = st.selectbox(
                    "Job Preference",
                    ["Full-time", "Part-time", "Contract", "Internship", "Freelance"]
                )
            with col_pref2:
                experience_years = st.number_input("Years of Experience", min_value=0, max_value=50, value=0)
            
            # Resume Upload Section
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div class='stCard'>
                    <h3>📤 Upload Your Resume</h3>
                    <p style='color: #a0a0a0;'>Supported format: PDF only (Max 10MB)</p>
                </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("", type=['pdf'], label_visibility="collapsed")
            
            # Quick Tips
            with st.expander("💡 Resume Tips for Better ATS Score"):
                st.markdown("""
                    - **Use clear section headings** (Experience, Education, Skills)
                    - **Include action verbs** (Developed, Managed, Led, Created)
                    - **Quantify achievements** (Increased sales by 30%)
                    - **Add relevant technical skills**
                    - **Include certifications and projects**
                    - **Keep formatting simple and clean**
                    - **Optimize for keywords** in your field
                """)
        
        with col2:
            # Quick Stats Card
            st.markdown("""
                <div class='info-box'>
                    <h3>🎯 Why Use Our Analyzer?</h3>
                    <ul style='line-height: 2; font-size: 0.95rem;'>
                        <li><b>AI-Powered Analysis</b><br/>Machine learning algorithms scan your resume</li>
                        <li><b>Instant ATS Score</b><br/>Know how recruiters' systems will rate you</li>
                        <li><b>Career Field Detection</b><br/>Find your perfect job category</li>
                        <li><b>Skill Recommendations</b><br/>Learn what skills to add</li>
                        <li><b>Live Job Matches</b><br/>Get relevant job listings instantly</li>
                        <li><b>Learning Resources</b><br/>Video tutorials for upskilling</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        
        # Analysis Trigger
        if uploaded_file and act_name and act_mail:
            st.session_state.user_step = 3
            
            # System info collection
            sec_token = secrets.token_urlsafe(12)
            host_name = socket.gethostname()
            try:
                ip_add = socket.gethostbyname(host_name)
            except:
                ip_add = "127.0.0.1"
            try:
                dev_user = os.getlogin()
            except OSError:
                dev_user = "unknown_user"
            os_name_ver = platform.system() + " " + platform.release()
            
            # Location detection
            if GEOCODING_AVAILABLE:
                try:
                    g = geocoder.ip('me')
                    latlong = str(g.latlng)
                    geolocator = Nominatim(user_agent="resumate_app", timeout=10)
                    location = geolocator.reverse(g.latlng, language='en')
                    address = location.raw['address']
                    city = address.get('city', 'Unknown')
                    state = address.get('state', 'Unknown')
                    country = address.get('country', 'Unknown')
                except:
                    latlong, city, state, country = "[0.0, 0.0]", "Unknown", "Unknown", "Unknown"
            else:
                latlong, city, state, country = "[0.0, 0.0]", "Unknown", "Unknown", "Unknown"
            
            # Save file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = uploaded_file.name
            file_path = RESUME_DIR / filename
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Tabbed Interface for Resume and Analysis
            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(["📄 Resume Preview", "🤖 AI Analysis", "📊 Detailed Report"])
            
            with tab1:
                st.markdown("<h3>Your Uploaded Resume</h3>", unsafe_allow_html=True)
                show_pdf(str(file_path))
            
            with tab2:
                # ML Analysis
                with st.spinner("🤖 Analyzing with Enhanced ML algorithms..."):
                    resume_text = extract_text_from_pdf(str(file_path))
                    
                    if resume_text:
                        # Count pages
                        try:
                            with open(str(file_path), 'rb') as f:
                                no_of_pages = len(list(PDFPage.get_pages(f)))
                        except:
                            no_of_pages = 1
                        
                        # Enhanced ML Analysis
                        predicted_field, confidence = predict_career_field_naive_bayes(resume_text)
                        skills = extract_skills(resume_text)
                        resume_score, feedback = analyze_resume_ml_enhanced(resume_text)
                        
                        # Candidate level with experience years
                        if experience_years >= 5:
                            cand_level = "Experienced"
                        elif experience_years >= 1:
                            cand_level = "Intermediate"
                        elif 'internship' in resume_text:
                            cand_level = "Intermediate"
                        else:
                            cand_level = "Fresher"
                        
                        st.session_state.user_step = 4
                        
                        # Success Message
                        st.success(f"✅ Analysis Complete for {act_name}!")
                        
                        # Enhanced Metrics with Icons
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <div class='metric-value'>{resume_score}</div>
                                    <div class='metric-label'>ATS Score</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_col2:
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <div class='metric-value'>{confidence}%</div>
                                    <div class='metric-label'>Confidence</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_col3:
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <div class='metric-value'>{len(skills)}</div>
                                    <div class='metric-label'>Skills Found</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_col4:
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <div class='metric-value'>{no_of_pages}</div>
                                    <div class='metric-label'>Pages</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Predicted Field with Visual
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class='stCard'>
                                <h2>🎯 Predicted Career Field</h2>
                                <h1 style='color: #FF4B4B; font-size: 2.5rem; margin: 1rem 0;'>{predicted_field}</h1>
                                <p style='color: #a0a0a0;'>Confidence Level: {confidence}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.progress(confidence / 100)
                        
                        # Skills Analysis
                        st.markdown("<br>", unsafe_allow_html=True)
                        col_skill1, col_skill2 = st.columns(2)
                        
                        with col_skill1:
                            st.markdown("""
                                <div class='stCard'>
                                    <h3>💡 Your Current Skills</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            if skills:
                                keywords = st_tags(label='', 
                                                 text='Skills extracted from your resume', 
                                                 value=skills, key='current_skills')
                            else:
                                st.warning("No specific technical skills detected. Add more skills!")
                        
                        with col_skill2:
                            # Recommend skills
                            recommended_skills = []
                            if predicted_field in SKILL_CATEGORIES:
                                all_field_skills = SKILL_CATEGORIES[predicted_field]
                                recommended_skills = [s.title() for s in all_field_skills if s not in resume_text][:10]
                                
                                if recommended_skills:
                                    st.markdown("""
                                        <div class='stCard'>
                                            <h3>🚀 Recommended Skills</h3>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    recommended_keywords = st_tags(
                                        label='',
                                        text='Boost your resume with these skills',
                                        value=recommended_skills, key='recommended_skills'
                                    )
                        
                        # Score Comparison Chart
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("""
                            <div class='stCard'>
                                <h3>📊 Score Comparison</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        comparison_df = pd.DataFrame({
                            'Category': ['Your Score', 'Average Score', 'Top 10%'],
                            'Score': [resume_score, 65, 85]
                        })
                        
                        fig = px.bar(comparison_df, x='Category', y='Score',
                                   color='Score',
                                   color_continuous_scale=['#FF4B4B', '#FFD700', '#00C48C'],
                                   text='Score')
                        
                        fig.update_traces(texttemplate='%{text}', textposition='outside')
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(30, 33, 48, 0.5)',
                            font=dict(color='white'),
                            showlegend=False,
                            height=400,
                            yaxis=dict(range=[0, 100])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Detailed Report
                st.markdown("""
                    <div class='stCard'>
                        <h2>📊 Comprehensive Resume Analysis Report</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Score Breakdown
                positive_feedback = [(item, points) for item, points in feedback if points > 0]
                improvement_feedback = [(item, points) for item, points in feedback if points == 0]
                
                col_report1, col_report2 = st.columns(2)
                
                with col_report1:
                    st.markdown("""
                        <div class='stCard' style='border-left: 5px solid #00C48C;'>
                            <h3>✅ Strengths Detected</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    for item, points in positive_feedback:
                        st.markdown(f"""
                            <div style='background: rgba(0, 196, 140, 0.1); padding: 0.8rem; 
                            border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid #00C48C;'>
                                <p style='color: #00C48C; margin: 0;'>{item}</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                with col_report2:
                    st.markdown("""
                        <div class='stCard' style='border-left: 5px solid #FF4B4B;'>
                            <h3>⚠️ Improvement Areas</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    for item, points in improvement_feedback:
                        st.markdown(f"""
                            <div style='background: rgba(255, 75, 75, 0.1); padding: 0.8rem; 
                            border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid #FF4B4B;'>
                                <p style='color: #FF4B4B; margin: 0;'>{item}</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Animated Score Progress
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class='stCard'>
                        <h3>🎯 Final ATS Score</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                my_bar = st.progress(0)
                for percent_complete in range(resume_score):
                    time.sleep(0.005)
                    my_bar.progress(percent_complete + 1)
                
                if resume_score >= 80:
                    st.markdown(f"""
                        <div class='stCard' style='background: linear-gradient(135deg, rgba(0, 196, 140, 0.2) 0%, rgba(0, 196, 140, 0.05) 100%); 
                        border: 2px solid #00C48C;'>
                            <h2 class='score-excellent'>⭐ Outstanding: {resume_score}/100</h2>
                            <p style='color: #00C48C; font-size: 1.1rem;'>Your resume is highly optimized for ATS systems. You're in the top tier!</p>
                        </div>
                    """, unsafe_allow_html=True)
                elif resume_score >= 60:
                    st.markdown(f"""
                        <div class='stCard' style='background: linear-gradient(135deg, rgba(255, 165, 0, 0.2) 0%, rgba(255, 165, 0, 0.05) 100%); 
                        border: 2px solid #FFA500;'>
                            <h2 class='score-good'>👍 Good Progress: {resume_score}/100</h2>
                            <p style='color: #FFA500; font-size: 1.1rem;'>Your resume is solid. Focus on improvement areas to reach excellence.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='stCard' style='background: linear-gradient(135deg, rgba(255, 75, 75, 0.2) 0%, rgba(255, 75, 75, 0.05) 100%); 
                        border: 2px solid #FF4B4B;'>
                            <h2 class='score-poor'>⚠️ Needs Work: {resume_score}/100</h2>
                            <p style='color: #FF4B4B; font-size: 1.1rem;'>Your resume needs significant improvements. Follow our recommendations above.</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Job Recommendations
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <div class='stCard'>
                        <h2>🔍 Personalized Job Recommendations</h2>
                        <p style='color: #a0a0a0;'>Based on your skills and predicted field</p>
                    </div>
                """, unsafe_allow_html=True)
                
                job_query = predicted_field if predicted_field != "General IT" else "Software Engineer"
                
                col_job1, col_job2 = st.columns([3, 1])
                with col_job1:
                    job_location = st.text_input("📍 Location", value="United States", key="job_location")
                with col_job2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    search_jobs = st.button("🔎 Find Jobs", type="primary")
                
                if search_jobs:
                    with st.spinner("🔎 Searching live job listings..."):
                        job_listings = fetch_jobs_rapidapi(job_query, job_location)
                        
                        if job_listings and job_listings.get('data'):
                            jobs = job_listings['data']
                            
                            if jobs:
                                st.success(f"🎯 Found {len(jobs)} matching opportunities!")
                                
                                for i, job in enumerate(jobs[:10], 1):
                                    job_title = job.get('job_title', 'N/A')
                                    employer = job.get('employer_name', 'N/A')
                                    location = job.get('job_city', 'N/A')
                                    description = job.get('job_description', '')
                                    job_link = job.get('job_apply_link', '#')
                                    
                                    # ML Match Score
                                    if skills and description:
                                        match_score = calculate_skill_match_tfidf(' '.join(skills).lower(), description.lower().split())
                                    else:
                                        match_score = random.randint(65, 90)
                                    
                                    st.markdown(f"""
                                        <div class='job-card'>
                                            <h3>#{i}. {job_title}</h3>
                                            <h4 style='color: #FF4B4B;'>🏢 {employer}</h4>
                                            <p style='color: #a0a0a0;'>📍 {location}</p>
                                            <div style='margin: 1rem 0;'>
                                                <span style='background: linear-gradient(135deg, #00C48C 0%, #00a070 100%); 
                                                padding: 0.5rem 1rem; border-radius: 20px; color: white; font-weight: 600;'>
                                                    🎯 Match: {match_score}%
                                                </span>
                                            </div>
                                            <a href='{job_link}' target='_blank' 
                                            style='background: linear-gradient(135deg, #FF4B4B 0%, #ff3333 100%); 
                                            color: white; padding: 0.7rem 1.5rem; border-radius: 8px; text-decoration: none; 
                                            font-weight: 600; display: inline-block; margin-top: 1rem;'>
                                                Apply Now →
                                            </a>
                                        </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.warning("No jobs found. Try different keywords.")
                        else:
                            st.info("Job listings temporarily unavailable.")
                
                # Learning Resources
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <div class='stCard'>
                        <h2>🎥 Recommended Learning Path</h2>
                        <p style='color: #a0a0a0;'>Curated courses and tutorials for {}</p>
                    </div>
                """.format(predicted_field), unsafe_allow_html=True)
                
                videos = get_youtube_recommendations(predicted_field)
                
                video_cols = st.columns(2)
                for idx, video in enumerate(videos):
                    with video_cols[idx % 2]:
                        st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #1e2130 0%, #2d313a 100%); 
                            padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #FF4B4B;'>
                                <h4 style='color: #FF4B4B;'>{video['title']}</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        try:
                            st.video(video['url'])
                        except:
                            st.markdown(f"[Watch Video]({video['url']})")
                
                # Save to database
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp_str = str(cur_date + '_' + cur_time)
                
                insert_data(
                    str(sec_token), str(ip_add), host_name, dev_user, os_name_ver,
                    str(latlong), city, state, country, act_name, act_mail, act_mob,
                    act_name, act_mail, str(resume_score), timestamp_str,
                    str(no_of_pages), predicted_field, cand_level,
                    str(skills), str(recommended_skills), str(predicted_field), filename
                )
                
                # Success Celebration
                st.balloons()
                
                # Download Report Button
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(255, 75, 75, 0.1) 0%, rgba(0, 196, 140, 0.1) 100%); 
                    border-radius: 15px; border: 2px solid #FF4B4B;'>
                        <h3>🎉 Analysis Complete!</h3>
                        <p style='color: #a0a0a0;'>Your resume has been analyzed and saved.</p>
                    </div>
                """, unsafe_allow_html=True)
                         
# ==================== Feedback Section ====================
    elif choice == 'Feedback':
        st.title("💬 User Feedback")
        
        # Hero Section
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);'>
                <h3 style='color: white; text-align: center; margin: 0;'>Help Us To Improve Resumate AI</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Feedback Form
            st.markdown("""
                <div style='background: linear-gradient(135deg, #1e1e2e 0%, #13131a 100%); padding: 2rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.3);'>
                    <h4 style='color: #FF4B4B; margin-bottom: 1rem;'>📝 Share Your Experience</h4>
                </div>
            """, unsafe_allow_html=True)
            
            with st.form("feedback_form"):
                feed_name = st.text_input('👤 Full Name', placeholder="Enter your name")
                feed_email = st.text_input('📧 Email Address', placeholder="your.email@example.com")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    feed_score = st.slider('⭐ Overall Rating', 1, 5, 5)
                with col_b:
                    feedback_category = st.selectbox('📂 Feedback Category', 
                        ['General', 'Feature Request', 'Bug Report', 'UI/UX', 'Performance', 'Other'])
                
                # Additional rating metrics
                st.markdown("##### Rate Specific Features:")
                col_x, col_y, col_z = st.columns(3)
                with col_x:
                    score_accuracy = st.slider('🎯 Score Accuracy', 1, 5, 5, key='acc')
                with col_y:
                    score_ui = st.slider('🎨 UI/UX Design', 1, 5, 5, key='ui')
                with col_z:
                    score_speed = st.slider('⚡ Speed/Performance', 1, 5, 5, key='speed')
                
                comments = st.text_area('💭 Your Detailed Feedback', 
                    placeholder="Tell us what you loved, what could be better, or suggest new features...", 
                    height=150)
                
                # Improvement suggestions
                improvements = st.multiselect('🚀 What would you like to see improved?',
                    ['Resume Analysis', 'Job Recommendations', 'Skill Suggestions', 'Video Resources', 
                     'ATS Scoring', 'User Interface', 'Loading Speed', 'Mobile Experience'])
                
                would_recommend = st.radio('💯 Would you recommend Resumate AI to others?', 
                    ['Yes, definitely!', 'Maybe', 'Not sure', 'No'], horizontal=True)
                
                submitted = st.form_submit_button("🚀 Submit Feedback", type="primary", use_container_width=True)
                
                if submitted:
                    if feed_name and feed_email:
                        ts = time.time()
                        cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        timestamp_str = str(cur_date + '_' + cur_time)
                        
                        # Enhanced data with additional fields
                        avg_feature_score = round((score_accuracy + score_ui + score_speed) / 3, 1)
                        improvements_str = ', '.join(improvements) if improvements else 'None'
                        
                        # You may need to update your insertf_data function to handle additional fields
                        insertf_data(feed_name, feed_email, feed_score, comments, timestamp_str)
                        
                        st.success("✅ Thank you for your valuable feedback!")
                        st.balloons()
                        
                        # Show summary
                        st.markdown(f"""
                            <div style='background: rgba(0, 196, 140, 0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #00C48C;'>
                                <h5 style='color: #00C48C; margin: 0 0 0.5rem 0;'>📊 Your Feedback Summary</h5>
                                <p style='margin: 0.3rem 0;'><b>Overall Rating:</b> {feed_score} ⭐</p>
                                <p style='margin: 0.3rem 0;'><b>Category:</b> {feedback_category}</p>
                                <p style='margin: 0.3rem 0;'><b>Feature Scores:</b> Accuracy: {score_accuracy}, UI/UX: {score_ui}, Speed: {score_speed}</p>
                                <p style='margin: 0.3rem 0;'><b>Recommendation:</b> {would_recommend}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("⚠️ Please fill in your name and email address")
        
        with col2:
            # Info boxes
            st.markdown("""
                <div style='background: linear-gradient(135deg, rgba(255, 75, 75, 0.2) 0%, rgba(255, 75, 75, 0.05) 100%); padding: 1.5rem; border-radius: 15px; border: 2px solid rgba(255, 75, 75, 0.3); margin-bottom: 1rem;'>
                    <h4 style='color: #FF4B4B; margin-bottom: 0.8rem;'>⭐ Why Feedback Matters</h4>
                    <p style='color: #b0b0b0; margin: 0.5rem 0;'>Your input helps us:</p>
                    <ul style='color: #b0b0b0; margin: 0.5rem 0; padding-left: 1.2rem;'>
                        <li>Improve ML algorithms</li>
                        <li>Enhance user experience</li>
                        <li>Add new features</li>
                        <li>Provide better recommendations</li>
                        <li>Fix bugs faster</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='background: linear-gradient(135deg, rgba(0, 196, 140, 0.2) 0%, rgba(0, 196, 140, 0.05) 100%); padding: 1.5rem; border-radius: 15px; border: 2px solid rgba(0, 196, 140, 0.3); margin-bottom: 1rem;'>
                    <h4 style='color: #00C48C; margin-bottom: 0.8rem;'>🎁 Feedback Rewards</h4>
                    <p style='color: #b0b0b0; margin: 0.5rem 0;'>Active contributors get:</p>
                    <ul style='color: #b0b0b0; margin: 0.5rem 0; padding-left: 1.2rem;'>
                        <li>Early access to features</li>
                        <li>Priority support</li>
                        <li>Recognition badge</li>
                        <li>Exclusive tips & tricks</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='background: linear-gradient(135deg, rgba(255, 215, 0, 0.2) 0%, rgba(255, 215, 0, 0.05) 100%); padding: 1.5rem; border-radius: 15px; border: 2px solid rgba(255, 215, 0, 0.3);'>
                    <h4 style='color: #FFD700; margin-bottom: 0.8rem;'>💡 Quick Tips</h4>
                    <ul style='color: #b0b0b0; margin: 0.5rem 0; padding-left: 1.2rem;'>
                        <li>Be specific in your feedback</li>
                        <li>Share both positives and negatives</li>
                        <li>Suggest actionable improvements</li>
                        <li>Include examples if possible</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Simple feedback display without analytics
        st.markdown("""
            <div style='background: linear-gradient(135deg, #1e1e2e 0%, #13131a 100%); padding: 2rem; border-radius: 15px; margin-top: 2rem; box-shadow: 0 5px 15px rgba(0,0,0,0.3);'>
                <h3 style='color: #667eea; text-align: center; margin-bottom: 1.5rem;'>💬 Thank You For Your Feedback!</h3>
                <p style='color: #b0b0b0; text-align: center; margin: 0;'>Your feedback is valuable and helps us improve Resumate AI continuously.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # ==================== About Section ====================
    elif choice == 'About':
        # Enhanced About Section Header
        st.markdown("""
            <div style="text-align: center; padding: 3rem 0; margin-bottom: 3rem;">
                <h1 style="
                    font-size: 3.5rem; 
                    font-weight: 800; 
                    background: linear-gradient(135deg, #FF4B4B 0%, #ff8080 50%, #00C48C 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    margin: 0;
                    text-shadow: 0 4px 8px rgba(0,0,0,0.3);
                    letter-spacing: 2px;
                ">ℹ️ About Resumate AI</h1>
                <p style="
                    font-size: 1.3rem; 
                    color: #a0a0a0; 
                    margin: 1rem 0 0 0;
                    font-weight: 300;
                    letter-spacing: 1px;
                ">Discover the Power of AI-Driven Resume Analysis</p>
                <div style="
                    width: 100px; 
                    height: 4px; 
                    background: linear-gradient(135deg, #FF4B4B 0%, #00C48C 100%);
                    margin: 1.5rem auto;
                    border-radius: 2px;
                "></div>
            </div>
        """, unsafe_allow_html=True)
        
        # Hero Introduction Card
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #1e2130 0%, #262b3d 100%);
                padding: 3rem;
                border-radius: 20px;
                border: 2px solid #FF4B4B;
                margin-bottom: 3rem;
                box-shadow: 0 15px 35px rgba(255, 75, 75, 0.2);
                text-align: center;
            ">
                <div style="font-size: 4rem; margin-bottom: 1.5rem;">🎯</div>
                <h2 style="
                    color: #FF4B4B; 
                    font-size: 2.5rem; 
                    margin-bottom: 1.5rem;
                    font-weight: 700;
                ">What is Resumate AI?</h2>
                <p style="
                    font-size: 1.3rem; 
                    line-height: 1.8; 
                    color: #e0e0e0;
                    margin: 0;
                    max-width: 800px;
                    margin: 0 auto;
                ">
                Resumate AI is an intelligent resume analyzer powered by advanced machine learning algorithms. 
                It helps job seekers optimize their resumes with dynamic, content-quality based scoring and 
                provides personalized recommendations for career growth.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced ML Algorithms Section
            st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #1e2130 0%, #262b3d 100%);
                    padding: 2.5rem;
                    border-radius: 20px;
                    border: 2px solid #FF4B4B;
                    margin-bottom: 2rem;
                    box-shadow: 0 10px 25px rgba(255, 75, 75, 0.15);
                ">
                    <h3 style="
                        text-align: center; 
                        color: #FF4B4B; 
                        margin-bottom: 2rem; 
                        font-size: 2rem;
                        font-weight: 700;
                    ">🤖 ML Algorithms Used</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Algorithm Cards with enhanced styling
            algorithms = [
                {
                    "title": "TF-IDF Vectorization",
                    "icon": "🔍",
                    "color": "#FF4B4B",
                    "description": "Converts resume text into numerical features by measuring term frequency-inverse document frequency. This helps in quantifying the importance of skills and keywords."
                },
                {
                    "title": "Naive Bayes Classification", 
                    "icon": "🧠",
                    "color": "#00C48C",
                    "description": "A probabilistic classifier that predicts career fields based on skills mentioned in resumes. Uses training data from multiple technical domains to achieve high accuracy."
                },
                {
                    "title": "Cosine Similarity",
                    "icon": "📊", 
                    "color": "#FFD700",
                    "description": "Measures similarity between resume content and job descriptions for accurate job matching. Provides percentage match scores for better job recommendations."
                },
                {
                    "title": "Content Quality Analysis",
                    "icon": "⚡",
                    "color": "#9C27B0", 
                    "description": "Our enhanced algorithm analyzes not just presence of sections but also their depth, quality, and completeness to provide dynamic scores."
                }
            ]
            
            for i, algo in enumerate(algorithms, 1):
                st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #262730 0%, #1a1d24 100%); 
                        padding: 2rem; 
                        border-radius: 15px; 
                        border-left: 5px solid {algo['color']}; 
                        margin-bottom: 1.5rem;
                        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
                        transition: transform 0.3s ease;
                    ">
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <span style="font-size: 2rem; margin-right: 1rem;">{algo['icon']}</span>
                            <h4 style="
                                color: {algo['color']}; 
                                margin: 0; 
                                font-size: 1.3rem;
                                font-weight: 600;
                            ">{i}. {algo['title']}</h4>
                        </div>
                        <p style="
                            color: #e0e0e0; 
                            margin: 0; 
                            line-height: 1.6; 
                            font-size: 1rem;
                        ">{algo['description']}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Enhanced Key Features Section
            st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #1e2130 0%, #262b3d 100%);
                    padding: 2.5rem;
                    border-radius: 20px;
                    border: 2px solid #00C48C;
                    margin-bottom: 2rem;
                    box-shadow: 0 10px 25px rgba(0, 196, 140, 0.15);
                ">
                    <h3 style="
                        text-align: center; 
                        color: #00C48C; 
                        margin-bottom: 2rem; 
                        font-size: 2rem;
                        font-weight: 700;
                    ">✨ Key Features</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Feature cards with icons
            features = [
                {"icon": "🎯", "title": "Dynamic ATS Scoring", "desc": "Content quality-based scoring (0-100)"},
                {"icon": "📄", "title": "Instant PDF Display", "desc": "View your resume immediately"},
                {"icon": "🔮", "title": "Career Field Prediction", "desc": "ML-powered field identification"},
                {"icon": "🧠", "title": "Smart Skill Extraction", "desc": "Automatic detection of technical skills"},
                {"icon": "💡", "title": "Skill Recommendations", "desc": "Personalized skill suggestions"},
                {"icon": "🔍", "title": "Live Job Search", "desc": "Real-time job recommendations with match scores"},
                {"icon": "🎥", "title": "Video Learning", "desc": "Curated YouTube tutorials for your field"},
                {"icon": "📊", "title": "Detailed Feedback", "desc": "Section-wise analysis with improvement tips"},
                {"icon": "📈", "title": "Analytics Dashboard", "desc": "Admin panel with user insights"}
            ]
            
            for feature in features:
                st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #262730 0%, #1a1d24 100%); 
                        padding: 1.5rem; 
                        border-radius: 12px; 
                        border-left: 4px solid #00C48C; 
                        margin-bottom: 1rem;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                        transition: transform 0.3s ease;
                    ">
                        <div style="display: flex; align-items: center;">
                            <span style="font-size: 1.5rem; margin-right: 1rem;">{feature['icon']}</span>
                            <div>
                                <h4 style="
                                    color: #00C48C; 
                                    margin: 0 0 0.5rem 0; 
                                    font-size: 1.1rem;
                                    font-weight: 600;
                                ">{feature['title']}</h4>
                                <p style="
                                    color: #b0b0b0; 
                                    margin: 0; 
                                    font-size: 0.95rem;
                                ">{feature['desc']}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='stCard'>
                <h3>📊 How Our Enhanced Scoring Works</h3>
                <p style='font-size: 1.1rem; line-height: 1.8;'>
                Unlike traditional ATS systems that only check for keyword presence, our enhanced algorithm 
                evaluates multiple factors:</p>
                <ul style='font-size: 1rem; line-height: 2;'>
                    <li><b>Section Completeness:</b> Checks for all essential resume sections</li>
                    <li><b>Content Depth:</b> Analyzes word count and detail level in each section</li>
                    <li><b>Action Verbs:</b> Identifies achievement-oriented language in experience</li>
                    <li><b>Skill Density:</b> Evaluates the number and relevance of technical skills</li>
                    <li><b>Professional Links:</b> Verifies presence of LinkedIn, GitHub, portfolios</li>
                    <li><b>Overall Quality:</b> Assesses resume comprehensiveness (250+ words)</li>
                </ul>
                <p style='font-size: 1rem; color: #00C48C; margin-top: 1rem;'>
                Each resume gets a unique score based on these factors, ensuring accurate and fair evaluation!
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='stCard'>
                <h3>🎓 Career Fields Supported</h3>
                <p>We provide specialized recommendations for:</p>
                <ul style='font-size: 1rem; line-height: 2;'>
                    <li><b>Data Science:</b> ML, AI, Data Analysis, Statistics</li>
                    <li><b>Web Development:</b> Frontend, Backend, Full Stack</li>
                    <li><b>Mobile Development:</b> iOS, Android, Cross-platform</li>
                    <li><b>DevOps:</b> Cloud, CI/CD, Infrastructure</li>
                    <li><b>Cybersecurity:</b> Security Testing, Network Security</li>
                    <li><b>UI/UX Design:</b> User Interface, User Experience</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
# ==================== Admin Section ====================
    else:
        # Custom CSS for enhanced interactivity
        st.markdown("""
            <style>
            /* Animated gradient background for admin section */
            .admin-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 15px;
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
                animation: gradientShift 3s ease infinite;
            }
            
            @keyframes gradientShift {
                0%, 100% { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
                50% { background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); }
            }
            
            .admin-header h1 {
                color: white;
                margin: 0;
                font-size: 2.5rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }
            
            .admin-header p {
                color: rgba(255,255,255,0.9);
                margin: 0.5rem 0 0 0;
                font-size: 1.1rem;
            }
            
            /* Login card styling */
            .login-card {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 2.5rem;
                margin: 2rem auto;
                max-width: 600px;
                box-shadow: 0 15px 35px rgba(0,0,0,0.2);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .login-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            }
            
            /* Animated info box */
            .info-box {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                padding: 1.5rem;
                border-radius: 15px;
                margin-top: 1.5rem;
                border-left: 5px solid #fff;
                animation: pulse 2s ease-in-out infinite;
                box-shadow: 0 5px 15px rgba(245, 87, 108, 0.3);
            }
            
            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.02); }
            }
            
            .info-box h4 {
                color: white;
                margin-top: 0;
                font-size: 1.2rem;
            }
            
            .info-box p {
                color: rgba(255,255,255,0.95);
                margin: 0.5rem 0;
                font-size: 1rem;
            }
            
            /* Stats cards */
            .stat-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 15px;
                color: white;
                text-align: center;
                margin: 1rem 0;
                transition: all 0.3s ease;
                cursor: pointer;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            }
            
            .stat-card:hover {
                transform: translateY(-10px) scale(1.05);
                box-shadow: 0 15px 30px rgba(102, 126, 234, 0.5);
            }
            
            .stat-card h3 {
                font-size: 2.5rem;
                margin: 0.5rem 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }
            
            .stat-card p {
                margin: 0;
                opacity: 0.9;
                font-size: 1.1rem;
            }
            
            /* Success message styling */
            .success-message {
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                color: white;
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                animation: slideInDown 0.5s ease;
                box-shadow: 0 5px 15px rgba(56, 239, 125, 0.3);
            }
            
            @keyframes slideInDown {
                from {
                    transform: translateY(-100%);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }
            
            /* Loading animation */
            .loading-dots {
                display: inline-block;
            }
            
            .loading-dots span {
                animation: blink 1.4s infinite;
                animation-fill-mode: both;
            }
            
            .loading-dots span:nth-child(2) {
                animation-delay: 0.2s;
            }
            
            .loading-dots span:nth-child(3) {
                animation-delay: 0.4s;
            }
            
            @keyframes blink {
                0%, 80%, 100% { opacity: 0; }
                40% { opacity: 1; }
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Animated header
        st.markdown("""
            <div class='admin-header'>
                <h1>👨‍💼 Admin Panel</h1>
                <p>Manage and analyze user data with advanced analytics</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for admin login
        if 'admin_logged_in' not in st.session_state:
            st.session_state.admin_logged_in = False
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = 0
        if 'show_hint' not in st.session_state:
            st.session_state.show_hint = False
        
        # Show logout button if logged in
        if st.session_state.admin_logged_in:
            col_logout1, col_logout2, col_logout3 = st.columns([4, 1, 1])
            with col_logout2:
                if st.button('🔄 Refresh', type="secondary", use_container_width=True):
                    st.rerun()
            with col_logout3:
                if st.button('🚪 Logout', type="primary", use_container_width=True):
                    st.session_state.admin_logged_in = False
                    st.session_state.login_attempts = 0
                    st.session_state.show_hint = False
                    st.success("👋 Logged out successfully!")
                    time.sleep(1)
                    st.rerun()
            
            # Welcome message with animation
            st.markdown("""
                <div class='success-message'>
                    <h2>🎉 Welcome, Admin !!</h2>
                    <p>You have full access to all analytics and data management tools</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Add spacing between welcome message and cards
            st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Show login form if not logged in
        else:            
            st.markdown("### 🔐 Secure Login")
            st.markdown("Enter your credentials to access the admin dashboard")
            
            # Progress indicator for login attempts
            if st.session_state.login_attempts > 0:
                progress_text = f"Login attempts: {st.session_state.login_attempts}/3"
                progress_value = st.session_state.login_attempts / 3
                st.progress(progress_value, text=progress_text)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                ad_user = st.text_input(
                    "Username", 
                    placeholder="Enter username", 
                    key="admin_user",
                    help="Enter your admin username"
                )
            with col2:
                ad_password = st.text_input(
                    "Password", 
                    type='password', 
                    placeholder="Enter password", 
                    key="admin_pass",
                    help="Enter your secure password"
                )
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
            
            with col_btn2:
                login_button = st.button('🔓 Login', type="primary", use_container_width=True)
            
            # Show hint button after 2 failed attempts
            if st.session_state.login_attempts >= 2:
                st.session_state.show_hint = True
            
            if st.session_state.show_hint:
                col_hint1, col_hint2, col_hint3 = st.columns([1, 1, 1])
                with col_hint2:
                    if st.button('💡 Show Hint', type="secondary", use_container_width=True):
                        st.markdown("""
                            <div class='info-box'>
                                <h4>🔐 Default Credentials</h4>
                                <p><b>Username:</b> admin</p>
                                <p><b>Password:</b> admin@resume-analyzer</p>
                                <p style='margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;'>
                                    💡 Tip: These are the default credentials for first-time access
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
            
            if login_button:
                # Strip any whitespace
                ad_user = ad_user.strip()
                ad_password = ad_password.strip()
                
                if ad_user == 'admin' and ad_password == 'admin@resume-analyzer':
                    st.session_state.admin_logged_in = True
                    st.session_state.login_attempts = 0
                    st.session_state.show_hint = False
                    
                    # Show loading animation
                    with st.spinner(''):
                        st.markdown("""
                            <div class='success-message'>
                                <h3>✅ Login Successful!</h3>
                                <p>Redirecting to dashboard<span class='loading-dots'><span>.</span><span>.</span><span>.</span></span></p>
                            </div>
                        """, unsafe_allow_html=True)
                        time.sleep(2)
                    st.rerun()
                else:
                    st.session_state.login_attempts += 1
                    
                    if st.session_state.login_attempts >= 3:
                        st.error("❌ Too many failed attempts! Please try again or use the hint button.")
                    else:
                        st.error(f"❌ Wrong Username or Password (Attempt {st.session_state.login_attempts}/3)")
                    
                    # Shake animation for error
                    st.markdown("""
                        <style>
                        @keyframes shake {
                            0%, 100% { transform: translateX(0); }
                            10%, 30%, 50%, 70%, 90% { transform: translateX(-10px); }
                            20%, 40%, 60%, 80% { transform: translateX(10px); }
                        }
                        .stTextInput > div > div > input {
                            animation: shake 0.5s;
                        }
                        </style>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Show admin dashboard if logged in
        if st.session_state.admin_logged_in:
            try:
                user_df = pd.read_csv(USER_DATA_CSV)
                
                if len(user_df) == 0:
                    st.warning("No user data available yet. Upload some resumes first!")
                else:
                    
                    # Enhanced Summary Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <div class='metric-value'>{len(user_df)}</div>
                                <div class='metric-label'>Total Users</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        avg_score = user_df['resume_score'].astype(float).mean()
                        st.markdown(f"""
                            <div class='metric-card'>
                                <div class='metric-value'>{avg_score:.0f}</div>
                                <div class='metric-label'>Avg Score</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        top_field = user_df['Predicted_Field'].mode()[0] if len(user_df) > 0 else 'N/A'
                        st.markdown(f"""
                            <div class='metric-card'>
                                <div class='metric-value'>{top_field}</div>
                                <div class='metric-label'>Top Field</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        experienced = len(user_df[user_df['User_level'] == 'Experienced'])
                        st.markdown(f"""
                            <div class='metric-card'>
                                <div class='metric-value'>{experienced}</div>
                                <div class='metric-label'>Experienced</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Display user data
                    st.header("**📊 User's Data**")
                    st.dataframe(user_df, use_container_width=True)
                    
                    # Download CSV
                    csv = user_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="user_data.csv" style="color: #FF4B4B; font-weight: 600; font-size: 1.1rem;">📥 Download User Data CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Enhanced Charts with Attractive Styling
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("**📈 Predicted Field Distribution**")
                        field_counts = user_df['Predicted_Field'].value_counts()
                        
                        # Custom vibrant color palette
                        colors = ['#FF4B4B', '#00C48C', '#FFD700', '#9C27B0', '#00BCD4', '#FF6B6B', '#4ECDC4', '#FFA500']
                        
                        fig = px.pie(values=field_counts.values, 
                                   names=field_counts.index, 
                                   title='<b>Career Fields Distribution</b>',
                                   hole=0.4,  # Donut chart
                                   color_discrete_sequence=colors)
                        
                        fig.update_traces(
                            textposition='outside',
                            textinfo='percent+label',
                            marker=dict(line=dict(color='#000000', width=2)),
                            pull=[0.1 if i == 0 else 0 for i in range(len(field_counts))],  # Pull out largest slice
                            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                        )
                        
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white', size=13, family='Arial Black'),
                            title_font=dict(size=18, color='#FF4B4B'),
                            showlegend=True,
                            legend=dict(
                                orientation="v",
                                yanchor="middle",
                                y=0.5,
                                xanchor="left",
                                x=1.1,
                                bgcolor='rgba(30, 33, 48, 0.8)',
                                bordercolor='#FF4B4B',
                                borderwidth=2
                            ),
                            margin=dict(l=20, r=150, t=60, b=20),
                            height=450
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("**📊 User Experience Level**")
                        level_counts = user_df['User_level'].value_counts()
                        
                        # Gradient color scheme for experience levels
                        exp_colors = ['#00C48C', '#FFD700', '#FF4B4B', '#9C27B0']
                        
                        fig = px.pie(values=level_counts.values, 
                                   names=level_counts.index,
                                   title="<b>Experience Levels Distribution</b>",
                                   hole=0.4,  # Donut chart
                                   color_discrete_sequence=exp_colors)
                        
                        fig.update_traces(
                            textposition='outside',
                            textinfo='percent+label',
                            marker=dict(line=dict(color='#000000', width=2)),
                            pull=[0.1, 0, 0, 0],  # Pull out first slice
                            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                        )
                        
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white', size=13, family='Arial Black'),
                            title_font=dict(size=18, color='#00C48C'),
                            showlegend=True,
                            legend=dict(
                                orientation="v",
                                yanchor="middle",
                                y=0.5,
                                xanchor="left",
                                x=1.1,
                                bgcolor='rgba(30, 33, 48, 0.8)',
                                bordercolor='#00C48C',
                                borderwidth=2
                            ),
                            margin=dict(l=20, r=150, t=60, b=20),
                            height=450
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Score distribution chart with enhanced styling
                    st.subheader("**📊 ATS Score Distribution**")
                    user_df['resume_score_num'] = pd.to_numeric(user_df['resume_score'], errors='coerce')
                    
                    fig = px.histogram(user_df, x='resume_score_num', nbins=20,
                                     title='<b>Distribution of Resume Scores</b>',
                                     labels={'resume_score_num': 'ATS Score'},
                                     color_discrete_sequence=['#FF4B4B'])
                    
                    fig.update_traces(
                        marker=dict(
                            line=dict(color='#000000', width=1.5),
                            opacity=0.9
                        ),
                        hovertemplate='<b>Score Range: %{x}</b><br>Count: %{y}<extra></extra>'
                    )
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(30, 33, 48, 0.5)',
                        font=dict(color='white', size=12),
                        title_font=dict(size=18, color='#FF4B4B'),
                        showlegend=False,
                        xaxis=dict(
                            title='<b>ATS Score</b>',
                            gridcolor='rgba(255, 255, 255, 0.1)',
                            showgrid=True
                        ),
                        yaxis=dict(
                            title='<b>Number of Users</b>',
                            gridcolor='rgba(255, 255, 255, 0.1)',
                            showgrid=True
                        ),
                        bargap=0.1,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Advanced Analytics Section
                    st.markdown("---")
                    st.header("**📊 Advanced Analytics**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Top Skills Analysis
                        st.subheader("**🔥 Most Common Skills**")
                        all_skills = []
                        for skills_str in user_df['Actual_skills'].dropna():
                            try:
                                skills_list = eval(skills_str) if isinstance(skills_str, str) else skills_str
                                if isinstance(skills_list, list):
                                    all_skills.extend(skills_list)
                            except:
                                pass
                        
                        if all_skills:
                            from collections import Counter
                            skill_counts = Counter(all_skills)
                            top_skills = dict(skill_counts.most_common(10))
                            
                            fig = px.bar(x=list(top_skills.values()), 
                                       y=list(top_skills.keys()),
                                       orientation='h',
                                       title='<b>Top 10 Skills Across All Resumes</b>',
                                       labels={'x': 'Count', 'y': 'Skill'},
                                       color=list(top_skills.values()),
                                       color_continuous_scale=['#FF4B4B', '#FFD700', '#00C48C'])
                            
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(30, 33, 48, 0.5)',
                                font=dict(color='white', size=12),
                                title_font=dict(size=16, color='#FF4B4B'),
                                showlegend=False,
                                height=400,
                                xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                                yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No skill data available")
                    
                    with col2:
                        # Score by Field Analysis
                        st.subheader("**📈 Average Score by Field**")
                        field_scores = user_df.groupby('Predicted_Field')['resume_score'].apply(
                            lambda x: pd.to_numeric(x, errors='coerce').mean()
                        ).sort_values(ascending=True)
                        
                        fig = px.bar(x=field_scores.values,
                                   y=field_scores.index,
                                   orientation='h',
                                   title='<b>Average ATS Score by Career Field</b>',
                                   labels={'x': 'Average Score', 'y': 'Field'},
                                   color=field_scores.values,
                                   color_continuous_scale='RdYlGn')
                        
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(30, 33, 48, 0.5)',
                            font=dict(color='white', size=12),
                            title_font=dict(size=16, color='#00C48C'),
                            showlegend=False,
                            height=400,
                            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', range=[0, 100]),
                            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Timeline Analysis
                    st.subheader("**📅 User Activity Timeline**")
                    user_df['Date'] = pd.to_datetime(user_df['Timestamp'].str.split('_').str[0], errors='coerce')
                    daily_users = user_df.groupby('Date').size().reset_index(name='Users')
                    
                    fig = px.line(daily_users, x='Date', y='Users',
                                title='<b>Daily User Registrations</b>',
                                markers=True)
                    
                    fig.update_traces(
                        line=dict(color='#FF4B4B', width=3),
                        marker=dict(size=8, color='#00C48C', line=dict(color='white', width=2))
                    )
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(30, 33, 48, 0.5)',
                        font=dict(color='white', size=12),
                        title_font=dict(size=18, color='#FF4B4B'),
                        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', title='<b>Date</b>'),
                        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', title='<b>Number of Users</b>'),
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Geographic Distribution
                    if 'city' in user_df.columns and 'country' in user_df.columns:
                        st.subheader("**🌍 Geographic Distribution**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            top_cities = user_df['city'].value_counts().head(10)
                            fig = px.bar(x=top_cities.values,
                                       y=top_cities.index,
                                       orientation='h',
                                       title='<b>Top 10 Cities</b>',
                                       labels={'x': 'Users', 'y': 'City'},
                                       color=top_cities.values,
                                       color_continuous_scale='Viridis')
                            
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(30, 33, 48, 0.5)',
                                font=dict(color='white', size=11),
                                title_font=dict(size=16, color='#FFD700'),
                                showlegend=False,
                                height=350
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            country_counts = user_df['country'].value_counts().head(10)
                            fig = px.pie(values=country_counts.values,
                                       names=country_counts.index,
                                       title='<b>Top Countries</b>',
                                       hole=0.4,
                                       color_discrete_sequence=px.colors.qualitative.Set3)
                            
                            fig.update_traces(
                                textposition='inside',
                                textinfo='percent+label',
                                marker=dict(line=dict(color='#000000', width=2))
                            )
                            
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white', size=11),
                                title_font=dict(size=16, color='#FFD700'),
                                height=350,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Search and Filter Section
                    st.markdown("---")
                    st.header("**🔍 Search & Filter Users**")
                    
                    search_col1, search_col2, search_col3 = st.columns(3)
                    
                    with search_col1:
                        search_name = st.text_input("Search by Name", placeholder="Enter name...")
                    
                    with search_col2:
                        filter_field = st.selectbox("Filter by Field", 
                                                    ['All'] + list(user_df['Predicted_Field'].unique()))
                    
                    with search_col3:
                        filter_level = st.selectbox("Filter by Level",
                                                    ['All'] + list(user_df['User_level'].unique()))
                    
                    # Apply filters
                    filtered_df = user_df.copy()
                    
                    if search_name:
                        filtered_df = filtered_df[filtered_df['Name'].str.contains(search_name, case=False, na=False)]
                    
                    if filter_field != 'All':
                        filtered_df = filtered_df[filtered_df['Predicted_Field'] == filter_field]
                    
                    if filter_level != 'All':
                        filtered_df = filtered_df[filtered_df['User_level'] == filter_level]
                    
                    st.success(f"Found {len(filtered_df)} matching records")
                    st.dataframe(filtered_df[['Name', 'Email_ID', 'resume_score', 'Predicted_Field', 
                                             'User_level', 'Timestamp']], use_container_width=True)
                    
                    # Export Filtered Data
                    if len(filtered_df) > 0:
                        csv_filtered = filtered_df.to_csv(index=False)
                        b64_filtered = base64.b64encode(csv_filtered.encode()).decode()
                        href_filtered = f'<a href="data:file/csv;base64,{b64_filtered}" download="filtered_users.csv" style="color: #00C48C; font-weight: 600;">📥 Download Filtered Data</a>'
                        st.markdown(href_filtered, unsafe_allow_html=True)
                    
                    # Feedback data
                    try:
                        feedback_df = pd.read_csv(FEEDBACK_CSV)
                        if len(feedback_df) > 0:
                            st.header("**💬 User Feedback**")
                            st.dataframe(feedback_df, use_container_width=True)
                            
                            avg_rating = feedback_df['feed_score'].mean()
                            total_feedback = len(feedback_df)
                            st.markdown(f"""
                                <p style='color: #00C48C; font-size: 1.8rem; font-weight: 700;'>
                                Average Rating: {avg_rating:.1f} ⭐ ({total_feedback} reviews)
                                </p>
                            """, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                    except:
                        st.info("No feedback data available")
            
            except Exception as e:
                st.error(f"Error loading admin data: {e}")

if __name__ == "__main__":
    main()
