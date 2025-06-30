python -m venv .venv
.\.venv\Scripts\activate          
pip install --upgrade pip wheel
pip install fastapi uvicorn[standard] python-jose[cryptography]
 
 
uvicorn touchcut_simulator:app --reload
