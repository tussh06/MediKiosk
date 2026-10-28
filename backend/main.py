import json
import os, uuid, re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from PIL import Image
import io
import pytesseract
import fitz
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, String, Boolean, DateTime, ForeignKey, Text, Integer, select, and_, or_
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medikiosk.db")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer(auto_error=False)

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="patient")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    doctor: Mapped[Optional["DoctorProfile"]] = relationship(back_populates="user", uselist=False)
    patient: Mapped[Optional["PatientProfile"]] = relationship(back_populates="user", uselist=False)

class UserStatus(Base):
    __tablename__ = "user_status"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

class DoctorProfile(Base):
    __tablename__ = "doctor_profiles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True)
    specialty: Mapped[str] = mapped_column(String(100))
    city: Mapped[str] = mapped_column(String(100), default="")
    bio: Mapped[str] = mapped_column(Text, default="")
    consultation_fee: Mapped[int] = mapped_column(Integer, default=0)
    consultation_type: Mapped[str] = mapped_column(String(30), default="online")
    experience_years: Mapped[int] = mapped_column(Integer, default=0)
    license_number: Mapped[str] = mapped_column(String(120), default="")
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    user: Mapped["User"] = relationship(back_populates="doctor")

class PatientProfile(Base):
    __tablename__ = "patient_profiles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True)
    phone: Mapped[str] = mapped_column(String(30), default="")
    dob: Mapped[str] = mapped_column(String(30), default="")
    blood_group: Mapped[str] = mapped_column(String(10), default="")
    allergies: Mapped[str] = mapped_column(Text, default="")
    conditions: Mapped[str] = mapped_column(Text, default="")
    user: Mapped["User"] = relationship(back_populates="patient")

class Appointment(Base):
    __tablename__ = "appointments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    doctor_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    scheduled_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    mode: Mapped[str] = mapped_column(String(30), default="online")
    reason: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(30), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sender_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    receiver_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class MedicalReport(Base):
    __tablename__ = "medical_reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    original_name: Mapped[str] = mapped_column(String(255))
    stored_name: Mapped[str] = mapped_column(String(255), unique=True)
    content_type: Mapped[str] = mapped_column(String(120), default="application/octet-stream")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class DoctorAvailability(Base):
    __tablename__ = "doctor_availability"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    weekday: Mapped[int] = mapped_column(Integer)
    start_time: Mapped[str] = mapped_column(String(5), default="09:00")
    end_time: Mapped[str] = mapped_column(String(5), default="17:00")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

class Notification(Base):
    __tablename__ = "notifications"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(160))
    body: Mapped[str] = mapped_column(Text, default="")
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(160))
    target: Mapped[str] = mapped_column(String(160), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))



class ClinicalCase(Base):
    __tablename__ = "clinical_cases"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    language: Mapped[str] = mapped_column(String(20), default="en-IN")
    mode: Mapped[str] = mapped_column(String(20), default="general")
    status: Mapped[str] = mapped_column(String(20), default="submitted")
    chief_complaint: Mapped[str] = mapped_column(Text, default="")
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class ConsentRecord(Base):
    __tablename__ = "consent_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    case_id: Mapped[Optional[int]] = mapped_column(ForeignKey("clinical_cases.id"), nullable=True)
    consent_type: Mapped[str] = mapped_column(String(80), default="clinical_intake")
    granted: Mapped[bool] = mapped_column(Boolean, default=False)
    language: Mapped[str] = mapped_column(String(20), default="en-IN")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

class DocumentInsight(Base):
    __tablename__ = "document_insights"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("medical_reports.id"), unique=True)
    extracted_text: Mapped[str] = mapped_column(Text, default="")
    diagnoses_json: Mapped[str] = mapped_column(Text, default="[]")
    medications_json: Mapped[str] = mapped_column(Text, default="[]")
    investigations_json: Mapped[str] = mapped_column(Text, default="[]")
    abnormal_json: Mapped[str] = mapped_column(Text, default="[]")
    document_date: Mapped[str] = mapped_column(String(30), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class IntegrationEvent(Base):
    __tablename__ = "integration_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    case_id: Mapped[Optional[int]] = mapped_column(ForeignKey("clinical_cases.id"), nullable=True)
    target: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(30), default="simulated")
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(engine)

def bootstrap_admin():
    email = os.getenv("ADMIN_EMAIL", "").strip().lower()
    password = os.getenv("ADMIN_PASSWORD", "")
    if not email or not password:
        return
    s = SessionLocal()
    try:
        existing = s.scalar(select(User).where(User.email == email))
        if not existing:
            admin = User(name="MediKiosk Admin", email=email, password_hash=pwd.hash(password), role="admin")
            s.add(admin); s.flush(); s.add(UserStatus(user_id=admin.id, active=True)); s.commit()
    finally:
        s.close()

bootstrap_admin()

app = FastAPI(title="MediKiosk API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: str = Field(pattern="^(patient|doctor)$")
    specialty: Optional[str] = ""
    city: Optional[str] = ""
    bio: Optional[str] = ""
    consultation_fee: int = Field(default=0, ge=0)
    consultation_type: str = "online"
    experience_years: int = Field(default=0, ge=0, le=80)
    license_number: Optional[str] = ""

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class PatientIn(BaseModel):
    phone: str = ""
    dob: str = ""
    blood_group: str = ""
    allergies: str = ""
    conditions: str = ""

class DoctorIn(BaseModel):
    specialty: str = Field(min_length=2, max_length=100)
    city: str = ""
    bio: str = ""
    consultation_fee: int = Field(default=0, ge=0)
    consultation_type: str = "online"
    experience_years: int = Field(default=0, ge=0, le=80)
    license_number: str = ""

class AppointmentIn(BaseModel):
    doctor_id: int
    scheduled_at: datetime
    mode: str = "online"
    reason: str = ""

class AppointmentUpdate(BaseModel):
    status: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    mode: Optional[str] = None
    reason: Optional[str] = None

class MessageIn(BaseModel):
    receiver_id: int
    body: str = Field(min_length=1, max_length=5000)

class AvailabilityIn(BaseModel):
    weekday: int = Field(ge=0, le=6)
    start_time: str = Field(pattern=r"^\d{2}:\d{2}$")
    end_time: str = Field(pattern=r"^\d{2}:\d{2}$")
    enabled: bool = True

class UserStatusIn(BaseModel):
    active: bool


class ClinicalCaseIn(BaseModel):
    language: str = "en-IN"
    mode: str = Field(default="general", pattern="^(general|ayush)$")
    answers: dict = {}
    consent_granted: bool = False
    abha_id: str = ""

class ConsentIn(BaseModel):
    consent_type: str = "clinical_intake"
    granted: bool = True
    language: str = "en-IN"


def _clean_list(values):
    return [x for x in values if x and str(x).strip()]

def _flatten_answers(answers):
    parts=[]
    for k,v in answers.items():
        if isinstance(v, list): v=", ".join(map(str,v))
        if v is not None and str(v).strip(): parts.append(f"{k}: {v}")
    return "\n".join(parts)

def _case_summary(answers, mode, language, abha_id=""):
    a={str(k): (", ".join(map(str,v)) if isinstance(v,list) else str(v or "").strip()) for k,v in answers.items()}
    chief=a.get("chief_complaint", "").strip()
    history={
        "chief_complaint": chief,
        "hpi": {k.replace("hpi_",""):v for k,v in a.items() if k.startswith("hpi_") and v},
        "past_medical_surgical": a.get("past_history", ""),
        "drug_and_allergy": {"medicines":a.get("current_medicines",""),"allergies":a.get("allergies","")},
        "family_history": a.get("family_history", ""),
        "personal_history": {k.replace("personal_",""):v for k,v in a.items() if k.startswith("personal_") and v},
        "review_of_systems": a.get("review_of_systems", ""),
        "prior_investigations": a.get("prior_investigations", ""),
    }
    ayush={k.replace("ayush_",""):v for k,v in a.items() if k.startswith("ayush_") and v}
    combined=_flatten_answers(a)
    risk=_risk_screen(combined)
    return {
        "format":"MediKiosk Clinical Intake v1",
        "language":language,"mode":mode,"abha_id":abha_id,
        "sections":history,"ayush_assessment":ayush,
        "red_flags":risk["matched_red_flags"],"risk_level":risk["risk_level"],"risk_score":risk["risk_score"],
        "risk_advice":risk["advice"],
        "physician_note":"Draft generated from patient-entered/voice answers. Physician must review, edit and confirm before clinical use."
    }

def _case_dict(c):
    data=json.loads(c.summary_json or "{}")
    return {"id":c.id,"patient_id":c.patient_id,"language":c.language,"mode":c.mode,"status":c.status,
            "chief_complaint":c.chief_complaint,"summary":data,"created_at":c.created_at.isoformat(),"submitted_at":c.submitted_at.isoformat() if c.submitted_at else None}

def _parse_document_insights(text):
    clean=re.sub(r"\s+", " ", text).strip()
    diagnoses=[]
    for pat in [r"(?:diagnosis|diagnoses|impression|assessment)\s*[:\-]\s*([^.;]+)", r"(?:diagnosed with|history of)\s+([^.;]+)"]:
        for m in re.finditer(pat, clean, re.I): diagnoses.append(m.group(1).strip())
    meds=[]
    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9\-]{2,30})\s+(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|mL|IU))\b(?:\s+(?:once|twice|thrice|daily|bd|od|tid|qid))?", clean, re.I):
        meds.append({"name":m.group(1),"dose":m.group(2)})
    investigations=[]
    abnormal=[]
    patterns={
        "hemoglobin":(r"(?:haemoglobin|hemoglobin|hb)\s*[:=]?\s*(\d+(?:\.\d+)?)",12,18,"g/dL"),
        "glucose":(r"(?:fasting\s+)?(?:blood\s+)?glucose\s*[:=]?\s*(\d+(?:\.\d+)?)",70,140,"mg/dL"),
        "spo2":(r"(?:spo2|oxygen\s+saturation)\s*[:=]?\s*(\d+(?:\.\d+)?)",95,100,"%"),
        "systolic_bp":(r"(?:bp|blood pressure)\s*[:=]?\s*(\d{2,3})\s*/\s*\d{2,3}",90,120,"mmHg"),
    }
    for name,(pat,lo,hi,unit) in patterns.items():
        for m in re.finditer(pat, clean, re.I):
            value=float(m.group(1)); item={"test":name,"value":value,"unit":unit,"reference":f"{lo}-{hi}"}; investigations.append(item)
            if value<lo or value>hi: abnormal.append({**item,"flag":"out_of_range"})
    dates=re.findall(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b", clean)
    return {"diagnoses":_clean_list(diagnoses)[:20],"medications":meds[:30],"investigations":investigations[:30],"abnormal_values":abnormal[:30],"document_dates":dates[:10]}

_INTERACTION_PAIRS=[("warfarin","aspirin","Potential bleeding risk; clinician review required."),("warfarin","ibuprofen","Potential bleeding risk; clinician review required."),("metformin","contrast","Renal/contrast-related precautions may be relevant; clinician review required."),("simvastatin","clarithromycin","Potential interaction; clinician/pharmacist review required.")]
def _drug_interactions(meds):
    names=[str(x.get("name","")).lower() for x in meds]
    out=[]
    for a,b,msg in _INTERACTION_PAIRS:
        if a in names and b in names: out.append({"pair":[a,b],"message":msg})
    return out

def db():
    s = SessionLocal()
    try: yield s
    finally: s.close()

def make_token(user):
    payload = {"sub": str(user.id), "role": user.role, "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def current_user(creds: HTTPAuthorizationCredentials = Depends(bearer), s: Session = Depends(db)):
    if not creds: raise HTTPException(401, "Authentication required")
    try:
        data = jwt.decode(creds.credentials, JWT_SECRET, algorithms=["HS256"]); uid = int(data["sub"])
    except (JWTError, ValueError, KeyError): raise HTTPException(401, "Invalid or expired token")
    user = s.get(User, uid)
    if not user: raise HTTPException(401, "User not found")
    status = s.scalar(select(UserStatus).where(UserStatus.user_id == uid))
    if status and not status.active: raise HTTPException(403, "Your account is disabled")
    return user

def role_required(role):
    def dep(user=Depends(current_user)):
        if user.role != role: raise HTTPException(403, f"{role} access required")
        return user
    return dep

def admin_required(user=Depends(current_user)):
    if user.role != "admin": raise HTTPException(403, "Admin access required")
    return user

def audit(s, user_id, action, target=""):
    s.add(AuditLog(user_id=user_id, action=action, target=target))

def notify(s, user_id, title, body):
    s.add(Notification(user_id=user_id, title=title, body=body))

def doctor_dict(user, d):
    return {"id": d.id, "user_id": user.id, "name": user.name, "email": user.email, "specialty": d.specialty, "city": d.city, "bio": d.bio,
            "consultation_fee": d.consultation_fee, "consultation_type": d.consultation_type, "experience_years": d.experience_years,
            "license_number": d.license_number, "verified": d.verified}

def patient_dict(p):
    return {"phone": p.phone, "dob": p.dob, "blood_group": p.blood_group, "allergies": p.allergies, "conditions": p.conditions}

def appointment_dict(a, s):
    p=s.get(User,a.patient_id); d=s.get(User,a.doctor_id)
    return {"id":a.id,"patient_id":a.patient_id,"patient_name":p.name if p else "Unknown","doctor_id":a.doctor_id,
            "doctor_name":d.name if d else "Unknown","scheduled_at":a.scheduled_at.isoformat(),"mode":a.mode,"reason":a.reason,
            "status":a.status,"created_at":a.created_at.isoformat() if a.created_at else None}


CASE_QUESTIONS = {
 "general": [
  {"id":"chief_complaint","section":"Presenting complaint","question":"What is the main problem or symptom you are experiencing?","type":"text","required":True},
  {"id":"hpi_onset","section":"History of present illness","question":"When did it start, and was the onset sudden or gradual?","type":"text"},
  {"id":"hpi_character","section":"History of present illness","question":"What does it feel like? Please describe the character or severity.","type":"text"},
  {"id":"hpi_site_radiation","section":"History of present illness","question":"Where is it located? Does it move or spread anywhere?","type":"text"},
  {"id":"hpi_aggravating_relieving","section":"History of present illness","question":"What makes it better or worse?","type":"text"},
  {"id":"hpi_associated","section":"History of present illness","question":"Do you have any other symptoms along with it?","type":"text"},
  {"id":"past_history","section":"Past history","question":"Any previous illnesses, hospital admissions, operations or major injuries?","type":"text"},
  {"id":"current_medicines","section":"Medicines & allergies","question":"What medicines, supplements or treatments are you currently taking?","type":"text"},
  {"id":"allergies","section":"Medicines & allergies","question":"Do you have any medicine, food or other allergies?","type":"text"},
  {"id":"family_history","section":"Family history","question":"Any important illnesses in close family members?","type":"text"},
  {"id":"personal_diet","section":"Personal history","question":"Tell us about your usual diet and daily routine.","type":"text"},
  {"id":"personal_sleep","section":"Personal history","question":"How is your sleep?","type":"text"},
  {"id":"personal_substances","section":"Personal history","question":"Do you use tobacco, alcohol or any other substances?","type":"text"},
  {"id":"review_of_systems","section":"Review of systems","question":"Any fever, weight change, weakness, breathing problems, chest symptoms, bowel/bladder changes, neurological symptoms or other concerns?","type":"text"},
  {"id":"prior_investigations","section":"Prior investigations","question":"What previous tests or reports should the doctor know about?","type":"text"},
 ],
 "ayush": [
  {"id":"chief_complaint","section":"Presenting complaint","question":"What is your main concern or symptom?","type":"text","required":True},
  {"id":"hpi_onset","section":"Present illness","question":"When did it begin and how has it changed?","type":"text"},
  {"id":"hpi_character","section":"Present illness","question":"Describe the symptom, severity and pattern.","type":"text"},
  {"id":"past_history","section":"Past history","question":"Any previous illnesses, surgery or long-term medicines?","type":"text"},
  {"id":"current_medicines","section":"Medicines & allergies","question":"Current medicines, supplements or AYUSH therapies?","type":"text"},
  {"id":"allergies","section":"Medicines & allergies","question":"Any known allergies or medicine reactions?","type":"text"},
  {"id":"family_history","section":"Family history","question":"Relevant family history?","type":"text"},
  {"id":"personal_diet","section":"Ahara-Vihara","question":"Describe your food habits, appetite and daily routine.","type":"text"},
  {"id":"personal_sleep","section":"Ahara-Vihara","question":"Describe sleep quality and timing.","type":"text"},
  {"id":"personal_substances","section":"Ahara-Vihara","question":"Any tobacco, alcohol or other substance use?","type":"text"},
  {"id":"review_of_systems","section":"Review of systems","question":"Any other symptoms or body-system concerns?","type":"text"},
  {"id":"ayush_prakriti","section":"Dashavidha Pariksha","question":"Prakriti: how would you describe your natural body constitution?","type":"text"},
  {"id":"ayush_vikriti","section":"Dashavidha Pariksha","question":"Vikriti: what changes or imbalance are you currently experiencing?","type":"text"},
  {"id":"ayush_sara","section":"Dashavidha Pariksha","question":"Sara: describe the apparent quality/strength of tissues as known to you.","type":"text"},
  {"id":"ayush_samhanana","section":"Dashavidha Pariksha","question":"Samhanana: describe body build/compactness.","type":"text"},
  {"id":"ayush_pramana","section":"Dashavidha Pariksha","question":"Pramana: height, weight or other available measurements?","type":"text"},
  {"id":"ayush_satmya","section":"Dashavidha Pariksha","question":"Satmya: foods, routines or habits that suit you well or poorly?","type":"text"},
  {"id":"ayush_sattva","section":"Dashavidha Pariksha","question":"Sattva: describe mental resilience/stress response in your own words.","type":"text"},
  {"id":"ayush_ahara_shakti","section":"Dashavidha Pariksha","question":"Ahara Shakti: appetite and digestion capacity?","type":"text"},
  {"id":"ayush_vyayama_shakti","section":"Dashavidha Pariksha","question":"Vyayama Shakti: usual exercise capacity/tolerance?","type":"text"},
  {"id":"ayush_vaya","section":"Dashavidha Pariksha","question":"Vaya: age/life stage and any age-related concerns?","type":"text"},
 ]
}

@app.get("/api/case/questions")
def case_questions(mode:str="general", language:str="en-IN", user=Depends(role_required("patient"))):
    if mode not in CASE_QUESTIONS: raise HTTPException(400,"Invalid intake mode")
    return {"language":language,"mode":mode,"questions":CASE_QUESTIONS[mode],"supported_languages":["en-IN","hi-IN","bn-IN","ta-IN","te-IN","mr-IN","gu-IN","kn-IN","ml-IN","pa-IN"]}

@app.post("/api/cases")
def create_case(data:ClinicalCaseIn,user=Depends(role_required("patient")),s:Session=Depends(db)):
    if not data.consent_granted: raise HTTPException(400,"Consent is required before clinical intake submission")
    if not data.answers.get("chief_complaint","").strip(): raise HTTPException(400,"Main complaint is required")
    summary=_case_summary(data.answers,data.mode,data.language,data.abha_id.strip())
    c=ClinicalCase(patient_id=user.id,language=data.language,mode=data.mode,chief_complaint=data.answers.get("chief_complaint","").strip(),summary_json=json.dumps(summary))
    s.add(c); s.flush()
    s.add(ConsentRecord(patient_id=user.id,case_id=c.id,consent_type="clinical_intake",granted=True,language=data.language))
    # Simulated interoperability events. Live ABDM/HIS requires institution credentials and endpoint configuration.
    payload={"resourceType":"Bundle","type":"document","case_id":c.id,"patient_id":user.id,"summary":summary}
    s.add(IntegrationEvent(patient_id=user.id,case_id=c.id,target="ABDM-FHIR-DEMO",status="simulated",payload_json=json.dumps(payload)))
    s.add(IntegrationEvent(patient_id=user.id,case_id=c.id,target="HIS-DEMO",status="simulated",payload_json=json.dumps({"case_id":c.id,"patient_id":user.id})))
    audit(s,user.id,"clinical_case_submitted",str(c.id))
    if summary["risk_level"]=="HIGH": notify(s,user.id,"Priority triage flag","Your intake contains a high-priority red-flag phrase. Please seek urgent clinical assessment if symptoms are happening now.")
    s.commit(); s.refresh(c)
    return _case_dict(c)

@app.get("/api/cases")
def my_cases(user=Depends(role_required("patient")),s:Session=Depends(db)):
    rows=s.scalars(select(ClinicalCase).where(ClinicalCase.patient_id==user.id).order_by(ClinicalCase.created_at.desc())).all()
    return [_case_dict(x) for x in rows]

@app.get("/api/cases/{case_id}")
def get_case(case_id:int,user=Depends(current_user),s:Session=Depends(db)):
    c=s.get(ClinicalCase,case_id)
    if not c: raise HTTPException(404,"Clinical case not found")
    allowed=user.role=="admin" or c.patient_id==user.id
    if user.role=="doctor": allowed=s.scalar(select(Appointment).where(Appointment.doctor_id==user.id,Appointment.patient_id==c.patient_id)) is not None
    if not allowed: raise HTTPException(403,"You do not have access to this clinical case")
    return _case_dict(c)

@app.get("/api/doctor/patients/{patient_id}/cases")
def doctor_patient_cases(patient_id:int,user=Depends(role_required("doctor")),s:Session=Depends(db)):
    linked=s.scalar(select(Appointment).where(Appointment.doctor_id==user.id,Appointment.patient_id==patient_id))
    if not linked: raise HTTPException(403,"Patient is not linked to your care")
    rows=s.scalars(select(ClinicalCase).where(ClinicalCase.patient_id==patient_id).order_by(ClinicalCase.created_at.desc())).all()
    return [_case_dict(x) for x in rows]

@app.post("/api/consent")
def save_consent(data:ConsentIn,user=Depends(role_required("patient")),s:Session=Depends(db)):
    row=ConsentRecord(patient_id=user.id,consent_type=data.consent_type,granted=data.granted,language=data.language)
    s.add(row); audit(s,user.id,"consent_recorded",data.consent_type); s.commit(); s.refresh(row)
    return {"id":row.id,"granted":row.granted,"consent_type":row.consent_type,"language":row.language,"created_at":row.created_at.isoformat()}

@app.get("/api/consent")
def get_consents(user=Depends(role_required("patient")),s:Session=Depends(db)):
    rows=s.scalars(select(ConsentRecord).where(ConsentRecord.patient_id==user.id).order_by(ConsentRecord.created_at.desc())).all()
    return [{"id":x.id,"granted":x.granted,"consent_type":x.consent_type,"language":x.language,"created_at":x.created_at.isoformat(),"revoked_at":x.revoked_at.isoformat() if x.revoked_at else None} for x in rows]

@app.post("/api/consent/{consent_id}/revoke")
def revoke_consent(consent_id:int,user=Depends(role_required("patient")),s:Session=Depends(db)):
    row=s.get(ConsentRecord,consent_id)
    if not row or row.patient_id!=user.id: raise HTTPException(404,"Consent record not found")
    row.granted=False; row.revoked_at=datetime.now(timezone.utc)
    audit(s,user.id,"consent_revoked",str(consent_id)); s.commit()
    return {"id":row.id,"granted":False,"revoked_at":row.revoked_at.isoformat()}

@app.get("/api/timeline")
def patient_timeline(user=Depends(role_required("patient")),s:Session=Depends(db)):
    events=[]
    for r in s.scalars(select(MedicalReport).where(MedicalReport.patient_id==user.id)).all(): events.append({"date":r.created_at.isoformat(),"type":"document","title":r.original_name,"detail":"Medical document uploaded"})
    for c in s.scalars(select(ClinicalCase).where(ClinicalCase.patient_id==user.id)).all(): events.append({"date":c.created_at.isoformat(),"type":"case","title":"Clinical intake submitted","detail":c.chief_complaint})
    for a in s.scalars(select(Appointment).where(Appointment.patient_id==user.id)).all(): events.append({"date":a.scheduled_at.isoformat(),"type":"appointment","title":"Appointment with doctor","detail":a.status})
    return sorted(events,key=lambda x:x["date"],reverse=True)

@app.get("/api/reports/{report_id}/insights")
def report_insights(report_id:int,user=Depends(current_user),s:Session=Depends(db)):
    r=s.get(MedicalReport,report_id)
    if not r: raise HTTPException(404,"Report not found")
    allowed=user.role=="admin" or r.patient_id==user.id
    if user.role=="doctor": allowed=s.scalar(select(Appointment).where(Appointment.doctor_id==user.id,Appointment.patient_id==r.patient_id)) is not None
    if not allowed: raise HTTPException(403,"You do not have access to this report")
    existing=s.scalar(select(DocumentInsight).where(DocumentInsight.report_id==r.id))
    if not existing:
        path=UPLOAD_DIR/r.stored_name
        if not path.exists(): raise HTTPException(404,"Stored file not found")
        try: extracted=_ocr_report_text(path,r.content_type)
        except Exception as e: raise HTTPException(500,"OCR could not process this report") from e
        parsed=_parse_document_insights(extracted)
        existing=DocumentInsight(report_id=r.id,extracted_text=extracted[:30000],diagnoses_json=json.dumps(parsed["diagnoses"]),medications_json=json.dumps(parsed["medications"]),investigations_json=json.dumps(parsed["investigations"]),abnormal_json=json.dumps(parsed["abnormal_values"]),document_date=(parsed["document_dates"] or [""])[0])
        s.add(existing); s.commit(); s.refresh(existing)
    meds=json.loads(existing.medications_json or "[]")
    return {"report":report_dict(r),"extracted_text":existing.extracted_text,"diagnoses":json.loads(existing.diagnoses_json or "[]"),"medications":meds,"investigations":json.loads(existing.investigations_json or "[]"),"abnormal_values":json.loads(existing.abnormal_json or "[]"),"document_date":existing.document_date,"potential_interactions":_drug_interactions(meds),"note":"Automated extraction is for physician attention and verification, not diagnosis or prescribing."}

@app.get("/api/cases/{case_id}/fhir")
def case_fhir(case_id:int,user=Depends(current_user),s:Session=Depends(db)):
    c=s.get(ClinicalCase,case_id)
    if not c: raise HTTPException(404,"Clinical case not found")
    allowed=user.role=="admin" or c.patient_id==user.id
    if user.role=="doctor": allowed=s.scalar(select(Appointment).where(Appointment.doctor_id==user.id,Appointment.patient_id==c.patient_id)) is not None
    if not allowed: raise HTTPException(403,"You do not have access to this case")
    u=s.get(User,c.patient_id); summary=json.loads(c.summary_json or "{}")
    entries=[{"resource":{"resourceType":"Patient","id":f"patient-{u.id}","name":[{"text":u.name}]}}]
    if c.chief_complaint: entries.append({"resource":{"resourceType":"Condition","id":f"case-{c.id}","code":{"text":c.chief_complaint},"note":[{"text":"Patient-entered clinical intake; clinician verification required."}]}})
    for x in summary.get("sections",{}).get("prior_investigations",{}).items() if isinstance(summary.get("sections",{}).get("prior_investigations"),dict) else []: pass
    return {"resourceType":"Bundle","type":"collection","entry":entries,"meta":{"profile":["http://hl7.org/fhir/R4"],"tag":[{"system":"https://medikiosk.local","code":"demo-interoperability"}]}}

@app.post("/api/cases/{case_id}/push-his")
def push_case_his(case_id:int,user=Depends(current_user),s:Session=Depends(db)):
    c=s.get(ClinicalCase,case_id)
    if not c: raise HTTPException(404,"Clinical case not found")
    if user.role=="doctor": allowed=s.scalar(select(Appointment).where(Appointment.doctor_id==user.id,Appointment.patient_id==c.patient_id)) is not None
    else: allowed=user.role=="admin" or c.patient_id==user.id
    if not allowed: raise HTTPException(403,"You do not have access to this case")
    ev=IntegrationEvent(patient_id=c.patient_id,case_id=c.id,target="HIS-DEMO",status="simulated",payload_json=json.dumps({"case_id":c.id,"message":"FHIR-ready clinical summary queued for configured hospital system"}))
    s.add(ev); audit(s,user.id,"his_push_simulated",str(c.id)); s.commit(); return {"status":"simulated","message":"Demo HIS push recorded. Configure a hospital FHIR endpoint for a live integration."}

@app.get("/api/health")
def health():
    return {"ok": True, "service": "MediKiosk API", "version": "2.0.0"}

@app.post("/api/auth/register")
def register(data: RegisterIn, s: Session = Depends(db)):
    email=data.email.lower()
    if s.scalar(select(User).where(User.email==email)): raise HTTPException(409,"Email already registered")
    user=User(name=data.name.strip(),email=email,password_hash=pwd.hash(data.password),role=data.role)
    s.add(user); s.flush(); s.add(UserStatus(user_id=user.id, active=True))
    if data.role=="doctor":
        if not data.specialty: raise HTTPException(400,"Doctor specialty is required")
        s.add(DoctorProfile(user_id=user.id,specialty=data.specialty.strip(),city=data.city or "",bio=data.bio or "",
            consultation_fee=data.consultation_fee,consultation_type=data.consultation_type,experience_years=data.experience_years,license_number=data.license_number or ""))
    else: s.add(PatientProfile(user_id=user.id))
    audit(s,user.id,"account_registered",data.role); s.commit()
    return {"token":make_token(user),"user":{"id":user.id,"name":user.name,"email":user.email,"role":user.role}}

@app.post("/api/auth/login")
def login(data: LoginIn, s: Session=Depends(db)):
    user=s.scalar(select(User).where(User.email==data.email.lower()))
    if not user or not pwd.verify(data.password,user.password_hash): raise HTTPException(401,"Invalid email or password")
    st=s.scalar(select(UserStatus).where(UserStatus.user_id==user.id))
    if st and not st.active: raise HTTPException(403,"Your account is disabled")
    audit(s,user.id,"login"); s.commit()
    return {"token":make_token(user),"user":{"id":user.id,"name":user.name,"email":user.email,"role":user.role}}

@app.get("/api/me")
def me(user=Depends(current_user)):
    result={"id":user.id,"name":user.name,"email":user.email,"role":user.role}
    if user.role=="doctor" and user.doctor: result["doctor"]=doctor_dict(user,user.doctor)
    if user.role=="patient" and user.patient: result["patient"]=patient_dict(user.patient)
    return result

@app.get("/api/doctors")
def doctors(q:str="",specialty:str="",city:str="",consultation_type:str="",s:Session=Depends(db)):
    stmt=select(User,DoctorProfile).join(DoctorProfile,DoctorProfile.user_id==User.id).where(User.role=="doctor",DoctorProfile.verified==True)
    if q:
        like=f"%{q}%"; stmt=stmt.where(or_(User.name.ilike(like),DoctorProfile.specialty.ilike(like),DoctorProfile.city.ilike(like)))
    if specialty: stmt=stmt.where(DoctorProfile.specialty==specialty)
    if city: stmt=stmt.where(DoctorProfile.city.ilike(f"%{city}%"))
    if consultation_type: stmt=stmt.where(DoctorProfile.consultation_type==consultation_type)
    return [doctor_dict(u,d) for u,d in s.execute(stmt.order_by(User.name)).all()]

@app.get("/api/doctors/{doctor_id}")
def doctor(doctor_id:int,s:Session=Depends(db)):
    d=s.get(DoctorProfile,doctor_id)
    if not d or not d.verified: raise HTTPException(404,"Verified doctor not found")
    u=s.get(User,d.user_id); return doctor_dict(u,d)

@app.put("/api/doctor/profile")
def update_doctor(data:DoctorIn,user=Depends(role_required("doctor")),s:Session=Depends(db)):
    d=user.doctor
    d.specialty=data.specialty; d.city=data.city; d.bio=data.bio; d.consultation_fee=data.consultation_fee
    d.consultation_type=data.consultation_type; d.experience_years=data.experience_years; d.license_number=data.license_number
    d.verified=False
    audit(s,user.id,"doctor_profile_updated",str(d.id)); s.commit()
    return doctor_dict(user,d)

@app.put("/api/patient/profile")
def update_patient(data:PatientIn,user=Depends(role_required("patient")),s:Session=Depends(db)):
    p=user.patient
    p.phone,p.dob,p.blood_group,p.allergies,p.conditions=data.phone,data.dob,data.blood_group,data.allergies,data.conditions
    audit(s,user.id,"patient_profile_updated",str(p.id)); s.commit(); return patient_dict(p)

@app.get("/api/dashboard")
def dashboard(user=Depends(current_user),s:Session=Depends(db)):
    if user.role=="patient":
        ap=s.scalars(select(Appointment).where(Appointment.patient_id==user.id).order_by(Appointment.scheduled_at.asc())).all()
        rp=s.scalars(select(MedicalReport).where(MedicalReport.patient_id==user.id)).all()
        unread=s.scalar(select(Notification).where(Notification.user_id==user.id,Notification.read==False).count()) if False else len(s.scalars(select(Notification).where(Notification.user_id==user.id,Notification.read==False)).all())
        return {"role":"patient","appointments":len(ap),"reports":len(rp),"upcoming":next((appointment_dict(a,s) for a in ap if a.scheduled_at>=datetime.now() and a.status in {"pending","accepted"}),None),"unread_notifications":unread}
    if user.role=="doctor":
        ap=s.scalars(select(Appointment).where(Appointment.doctor_id==user.id)).all()
        pending=len([a for a in ap if a.status=="pending"])
        patients=len(set(a.patient_id for a in ap))
        unread=len(s.scalars(select(Notification).where(Notification.user_id==user.id,Notification.read==False)).all())
        return {"role":"doctor","appointments":len(ap),"pending":pending,"patients":patients,"unread_notifications":unread}
    users=len(s.scalars(select(User)).all()); docs=len(s.scalars(select(DoctorProfile)).all())
    aps=len(s.scalars(select(Appointment)).all()); msgs=len(s.scalars(select(Message)).all())
    pending=len(s.scalars(select(DoctorProfile).where(DoctorProfile.verified==False)).all())
    return {"role":"admin","users":users,"doctors":docs,"patients":len(s.scalars(select(User).where(User.role=="patient")).all()),"appointments":aps,"messages":msgs,"pending_doctors":pending}

@app.post("/api/appointments")
def book(data:AppointmentIn,user=Depends(role_required("patient")),s:Session=Depends(db)):
    doctor=s.get(User,data.doctor_id)
    if not doctor or doctor.role!="doctor" or not doctor.doctor or not doctor.doctor.verified: raise HTTPException(404,"Verified doctor not found")
    when=data.scheduled_at.replace(tzinfo=None)
    if when <= datetime.now(): raise HTTPException(400,"Appointment time must be in the future")
    conflict=s.scalar(select(Appointment).where(and_(Appointment.doctor_id==doctor.id,Appointment.scheduled_at==when,Appointment.status.in_(["pending","accepted"]))))
    if conflict: raise HTTPException(409,"That appointment slot is already booked")
    a=Appointment(patient_id=user.id,doctor_id=doctor.id,scheduled_at=when,mode=data.mode,reason=data.reason.strip())
    s.add(a); notify(s,doctor.id,"New appointment request",f"{user.name} requested an appointment."); audit(s,user.id,"appointment_created",str(a.id)); s.commit(); s.refresh(a)
    return appointment_dict(a,s)

@app.get("/api/appointments")
def appointments(user=Depends(current_user),s:Session=Depends(db)):
    stmt=select(Appointment).where(or_(Appointment.patient_id==user.id,Appointment.doctor_id==user.id)).order_by(Appointment.scheduled_at.desc())
    return [appointment_dict(a,s) for a in s.scalars(stmt).all()]

@app.patch("/api/appointments/{appointment_id}")
def update_appointment(appointment_id:int,data:AppointmentUpdate,user=Depends(current_user),s:Session=Depends(db)):
    a=s.get(Appointment,appointment_id)
    if not a: raise HTTPException(404,"Appointment not found")
    if user.id not in {a.patient_id,a.doctor_id}: raise HTTPException(403,"Not your appointment")
    if data.status and data.status not in {"accepted","declined","cancelled","completed"}: raise HTTPException(400,"Invalid status")
    if user.id==a.patient_id and data.status in {"accepted","declined","completed"}: raise HTTPException(403,"Patients cannot set that status")
    if user.id==a.doctor_id and data.status=="cancelled": pass
    if data.scheduled_at:
        when=data.scheduled_at.replace(tzinfo=None)
        if when<=datetime.now(): raise HTTPException(400,"New time must be in the future")
        conflict=s.scalar(select(Appointment).where(and_(Appointment.doctor_id==a.doctor_id,Appointment.id!=a.id,Appointment.scheduled_at==when,Appointment.status.in_(["pending","accepted"]))))
        if conflict: raise HTTPException(409,"That appointment slot is already booked")
        a.scheduled_at=when
    if data.mode: a.mode=data.mode
    if data.reason is not None: a.reason=data.reason
    if data.status:
        a.status=data.status
        other=a.patient_id if user.id==a.doctor_id else a.doctor_id
        notify(s,other,"Appointment updated",f"Appointment #{a.id} is now {a.status}.")
    audit(s,user.id,"appointment_updated",str(a.id)); s.commit()
    return appointment_dict(a,s)

@app.get("/api/doctor/patients")
def doctor_patients(user=Depends(role_required("doctor")),s:Session=Depends(db)):
    ids=s.scalars(select(Appointment.patient_id).where(Appointment.doctor_id==user.id)).all()
    ids=list(dict.fromkeys(ids))
    result=[]
    for uid in ids:
        u=s.get(User,uid); p=u.patient if u else None
        if u: result.append({"id":u.id,"name":u.name,"email":u.email,"profile":patient_dict(p) if p else {}})
    return result

@app.get("/api/doctor/patients/{patient_id}")
def doctor_patient(patient_id:int,user=Depends(role_required("doctor")),s:Session=Depends(db)):
    linked=s.scalar(select(Appointment).where(Appointment.doctor_id==user.id,Appointment.patient_id==patient_id))
    if not linked: raise HTTPException(403,"Patient is not linked to your care")
    u=s.get(User,patient_id)
    if not u or u.role!="patient": raise HTTPException(404,"Patient not found")
    reports=s.scalars(select(MedicalReport).where(MedicalReport.patient_id==patient_id).order_by(MedicalReport.created_at.desc())).all()
    return {"id":u.id,"name":u.name,"email":u.email,"profile":patient_dict(u.patient) if u.patient else {}, "reports":[report_dict(r) for r in reports]}

def _ocr_report_text(path: Path, content_type: str, lang: str = "eng+hin+ben+tam+tel+mar+guj+kan+mal+pan") -> str:
    """Extract text locally from an image or PDF using Tesseract OCR."""
    if content_type == "application/pdf":
        doc = fitz.open(path)
        chunks = []
        for page in doc:
            text = page.get_text("text").strip()
            if text:
                chunks.append(text)
            else:
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                chunks.append(pytesseract.image_to_string(img, lang=lang))
        doc.close()
        return "\n".join(chunks).strip()
    img = Image.open(path)
    return pytesseract.image_to_string(img, lang=lang).strip()

# This is a conservative screening aid, not a diagnosis or medical emergency detector.
_EMERGENCY_TERMS = {
    "chest pain": 40, "difficulty breathing": 40, "shortness of breath": 35,
    "severe bleeding": 45, "unconscious": 50, "loss of consciousness": 50,
    "stroke": 50, "seizure": 40, "severe allergic reaction": 45,
    "anaphylaxis": 50, "heart attack": 50, "myocardial infarction": 50,
}
_URGENT_TERMS = {
    "high fever": 15, "persistent vomiting": 15, "dehydration": 15,
    "blood in stool": 20, "vomiting blood": 30, "severe abdominal pain": 25,
    "very high blood pressure": 25, "oxygen saturation": 5,
}

def _risk_screen(text: str):
    t = re.sub(r"\s+", " ", text.lower()).strip()
    score = 0
    matched = []
    for term, pts in _EMERGENCY_TERMS.items():
        if term in t:
            score += pts; matched.append(term)
    for term, pts in _URGENT_TERMS.items():
        if term in t:
            score += pts; matched.append(term)
    score = min(score, 100)
    if score >= 40:
        level = "HIGH"
        advice = "Urgent clinical review is recommended. If these symptoms are happening now or are severe, contact local emergency medical services immediately."
    elif score >= 15:
        level = "MODERATE"
        advice = "Prompt review by a qualified healthcare professional is recommended, especially if symptoms are worsening."
    else:
        level = "LOW"
        advice = "No high-priority red-flag phrase was detected in the extracted text. This does not mean the patient is medically safe or disease-free."
    return {"risk_score": score, "risk_level": level, "matched_red_flags": matched,
            "advice": advice,
            "disclaimer": "OCR and rule-based screening can miss or misread medical information. This result is informational and must not be used as a diagnosis or to delay emergency care."}

def report_dict(r):
    return {"id":r.id,"name":r.original_name,"content_type":r.content_type,"created_at":r.created_at.isoformat()}

@app.post("/api/reports")
def upload_report(file:UploadFile=File(...),user=Depends(role_required("patient")),s:Session=Depends(db)):
    allowed={"application/pdf","image/png","image/jpeg","image/webp"}
    if file.content_type not in allowed: raise HTTPException(400,"Only PDF, PNG, JPG or WEBP files are allowed")
    data=file.file.read()
    if len(data)>10*1024*1024: raise HTTPException(413,"File too large (10 MB maximum)")
    original=Path(file.filename or "report").name
    stored=f"{uuid.uuid4().hex}{Path(original).suffix.lower()}"
    (UPLOAD_DIR/stored).write_bytes(data)
    r=MedicalReport(patient_id=user.id,original_name=original,stored_name=stored,content_type=file.content_type)
    s.add(r); audit(s,user.id,"report_uploaded",original); s.commit(); s.refresh(r)
    return report_dict(r)

@app.get("/api/reports/{report_id}/analysis")
def analyze_report(report_id:int,user=Depends(current_user),s:Session=Depends(db)):
    r=s.get(MedicalReport,report_id)
    if not r: raise HTTPException(404,"Report not found")
    allowed=user.role=="admin" or r.patient_id==user.id
    if user.role=="doctor":
        allowed=s.scalar(select(Appointment).where(Appointment.doctor_id==user.id,Appointment.patient_id==r.patient_id)) is not None
    if not allowed: raise HTTPException(403,"You do not have access to this report")
    path=UPLOAD_DIR/r.stored_name
    if not path.exists(): raise HTTPException(404,"Stored file not found")
    try:
        text=_ocr_report_text(path,r.content_type)
    except Exception as e:
        raise HTTPException(500,"OCR could not process this report") from e
    if not text:
        raise HTTPException(422,"No readable text was detected in this report")
    result=_risk_screen(text)
    return {"report":report_dict(r),"extracted_text":text[:12000],**result}

@app.get("/api/reports")
def reports(user=Depends(role_required("patient")),s:Session=Depends(db)):
    rows=s.scalars(select(MedicalReport).where(MedicalReport.patient_id==user.id).order_by(MedicalReport.created_at.desc())).all()
    return [report_dict(r) for r in rows]

@app.get("/api/reports/{report_id}/download")
def download_report(report_id:int,user=Depends(current_user),s:Session=Depends(db)):
    r=s.get(MedicalReport,report_id)
    if not r: raise HTTPException(404,"Report not found")
    allowed=user.role=="admin" or r.patient_id==user.id
    if user.role=="doctor":
        allowed=s.scalar(select(Appointment).where(Appointment.doctor_id==user.id,Appointment.patient_id==r.patient_id)) is not None
    if not allowed: raise HTTPException(403,"You do not have access to this report")
    path=UPLOAD_DIR/r.stored_name
    if not path.exists(): raise HTTPException(404,"Stored file not found")
    return FileResponse(path,media_type=r.content_type,filename=r.original_name)

@app.get("/api/doctor/availability")
def get_availability(user=Depends(role_required("doctor")),s:Session=Depends(db)):
    rows=s.scalars(select(DoctorAvailability).where(DoctorAvailability.doctor_id==user.id).order_by(DoctorAvailability.weekday)).all()
    return [{"id":x.id,"weekday":x.weekday,"start_time":x.start_time,"end_time":x.end_time,"enabled":x.enabled} for x in rows]

@app.put("/api/doctor/availability")
def set_availability(data:AvailabilityIn,user=Depends(role_required("doctor")),s:Session=Depends(db)):
    row=s.scalar(select(DoctorAvailability).where(DoctorAvailability.doctor_id==user.id,DoctorAvailability.weekday==data.weekday))
    if row: row.start_time=data.start_time; row.end_time=data.end_time; row.enabled=data.enabled
    else: s.add(DoctorAvailability(doctor_id=user.id,weekday=data.weekday,start_time=data.start_time,end_time=data.end_time,enabled=data.enabled))
    s.commit(); return {"ok":True}

@app.get("/api/messages/contacts")
def message_contacts(user=Depends(current_user),s:Session=Depends(db)):
    rows=s.scalars(select(Message).where(or_(Message.sender_id==user.id,Message.receiver_id==user.id)).order_by(Message.created_at.desc())).all()
    ids=[]
    for m in rows:
        other=m.receiver_id if m.sender_id==user.id else m.sender_id
        if other not in ids: ids.append(other)
    result=[]
    for uid in ids:
        u=s.get(User,uid)
        if u and u.role in {"doctor","patient"}: result.append({"id":u.id,"name":u.name,"role":u.role,"email":u.email})
    return result

@app.get("/api/messages")
def messages(user=Depends(current_user),with_user:int=Query(...),s:Session=Depends(db)):
    other=s.get(User,with_user)
    if not other or other.role not in {"doctor","patient"}: raise HTTPException(404,"User not found")
    stmt=select(Message).where(or_(and_(Message.sender_id==user.id,Message.receiver_id==with_user),and_(Message.sender_id==with_user,Message.receiver_id==user.id))).order_by(Message.created_at.asc())
    return [{"id":m.id,"sender_id":m.sender_id,"receiver_id":m.receiver_id,"body":m.body,"created_at":m.created_at.isoformat()} for m in s.scalars(stmt).all()]

@app.post("/api/messages")
def send_message(data:MessageIn,user=Depends(current_user),s:Session=Depends(db)):
    receiver=s.get(User,data.receiver_id)
    if not receiver or receiver.role not in {"doctor","patient"} or receiver.id==user.id: raise HTTPException(404,"User not found")
    m=Message(sender_id=user.id,receiver_id=receiver.id,body=data.body.strip())
    s.add(m); notify(s,receiver.id,"New message",f"{user.name} sent you a message."); audit(s,user.id,"message_sent",str(receiver.id)); s.commit(); s.refresh(m)
    return {"id":m.id,"sender_id":m.sender_id,"receiver_id":m.receiver_id,"body":m.body,"created_at":m.created_at.isoformat()}

@app.get("/api/notifications")
def notifications(user=Depends(current_user),s:Session=Depends(db)):
    rows=s.scalars(select(Notification).where(Notification.user_id==user.id).order_by(Notification.created_at.desc()).limit(30)).all()
    return [{"id":n.id,"title":n.title,"body":n.body,"read":n.read,"created_at":n.created_at.isoformat()} for n in rows]

@app.patch("/api/notifications/{notification_id}/read")
def read_notification(notification_id:int,user=Depends(current_user),s:Session=Depends(db)):
    n=s.get(Notification,notification_id)
    if not n or n.user_id!=user.id: raise HTTPException(404,"Notification not found")
    n.read=True; s.commit(); return {"ok":True}

@app.post("/api/notifications/read-all")
def read_all(user=Depends(current_user),s:Session=Depends(db)):
    rows=s.scalars(select(Notification).where(Notification.user_id==user.id,Notification.read==False)).all()
    for n in rows: n.read=True
    s.commit(); return {"ok":True}

@app.get("/api/admin/stats")
def admin_stats(user=Depends(admin_required),s:Session=Depends(db)):
    return dashboard(user,s)

@app.get("/api/admin/users")
def admin_users(user=Depends(admin_required),s:Session=Depends(db)):
    rows=s.scalars(select(User).order_by(User.created_at.desc())).all()
    result=[]
    for u in rows:
        st=s.scalar(select(UserStatus).where(UserStatus.user_id==u.id))
        result.append({"id":u.id,"name":u.name,"email":u.email,"role":u.role,"active":st.active if st else True,"created_at":u.created_at.isoformat()})
    return result

@app.patch("/api/admin/users/{user_id}/status")
def admin_user_status(user_id:int,data:UserStatusIn,user=Depends(admin_required),s:Session=Depends(db)):
    if user_id==user.id: raise HTTPException(400,"You cannot disable your own admin account")
    u=s.get(User,user_id)
    if not u: raise HTTPException(404,"User not found")
    st=s.scalar(select(UserStatus).where(UserStatus.user_id==user_id))
    if not st: st=UserStatus(user_id=user_id,active=data.active); s.add(st)
    else: st.active=data.active
    audit(s,user.id,"user_status_changed",f"{user_id}:{data.active}"); s.commit()
    return {"id":user_id,"active":data.active}

@app.get("/api/admin/doctors")
def admin_doctors(user=Depends(admin_required),s:Session=Depends(db)):
    stmt=select(User,DoctorProfile).join(DoctorProfile,DoctorProfile.user_id==User.id).where(User.role=="doctor").order_by(User.created_at.desc())
    return [doctor_dict(u,d) for u,d in s.execute(stmt).all()]

@app.post("/api/admin/verify-doctor/{doctor_id}")
def verify_doctor(doctor_id:int,user=Depends(admin_required),s:Session=Depends(db)):
    d=s.get(DoctorProfile,doctor_id)
    if not d: raise HTTPException(404,"Doctor not found")
    d.verified=True; notify(s,d.user_id,"Doctor verified","Your MediKiosk professional profile has been verified."); audit(s,user.id,"doctor_verified",str(doctor_id)); s.commit()
    return {"verified":True,"doctor_id":doctor_id}

@app.post("/api/admin/reject-doctor/{doctor_id}")
def reject_doctor(doctor_id:int,user=Depends(admin_required),s:Session=Depends(db)):
    d=s.get(DoctorProfile,doctor_id)
    if not d: raise HTTPException(404,"Doctor not found")
    d.verified=False; notify(s,d.user_id,"Verification update","Your doctor profile needs review before it can appear in the public directory."); audit(s,user.id,"doctor_rejected",str(doctor_id)); s.commit()
    return {"verified":False,"doctor_id":doctor_id}

@app.get("/api/admin/appointments")
def admin_appointments(user=Depends(admin_required),s:Session=Depends(db)):
    rows=s.scalars(select(Appointment).order_by(Appointment.scheduled_at.desc())).all()
    return [appointment_dict(a,s) for a in rows]

@app.get("/api/admin/reports")
def admin_reports(user=Depends(admin_required),s:Session=Depends(db)):
    rows=s.scalars(select(MedicalReport).order_by(MedicalReport.created_at.desc()).limit(100)).all()
    return [report_dict(r)|{"patient_id":r.patient_id} for r in rows]

@app.get("/api/admin/audit")
def admin_audit(user=Depends(admin_required),s:Session=Depends(db)):
    rows=s.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(100)).all()
    return [{"id":x.id,"user_id":x.user_id,"action":x.action,"target":x.target,"created_at":x.created_at.isoformat()} for x in rows]
