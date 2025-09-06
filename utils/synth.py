import numpy as np, pandas as pd, random, string
from datetime import datetime, timedelta

KENYAN_SURNAMES = [
    "Ochieng","Odhiambo","Otieno","Wanjiru","Wambui","Chebet","Mutiso","Mwangi","Koech","Kipchoge",
    "Njoroge","Kamau","Achieng","Cherono","Kiplagat","Kimutai","Korir","Chepngeno","Makena","Barasa",
    "Nyambura","Njoki","Kiptoo","Kipruto","Cheruiyot","Wairimu","Muthoni","Atieno","Wafula","Were",
]
ENGLISH_FIRST = [
    "Mary","John","Grace","Peter","Elizabeth","Kevin","Faith","James","Ann","Joseph","Irene","Paul",
    "Daniel","Sarah","David","Mercy","Esther","Samuel","Cynthia","Michael","Alice","Brian","Ivy","George"
]

OCCUPATIONS = [
    "mama mboga","shop owner","boda boda","farmer","salonist","tailor","carpenter","teacher","mechanic","hawker"
]
# Updated products to match real loan book
PRODUCTS = ["INUKA 4 WEEKS", "KUZA 4 WEEKS", "FADHILI 4 WEEKS", "INUKA 8 WEEKS", "KUZA 8 WEEKS", "BIASHARA 12 WEEKS", "EMERGENCY LOAN", "GROUP LOAN"]
LOAN_TYPES = ["Normal", "Group", "Emergency", "Salary Advance"]
LOAN_HEALTH = ["Performing", "Watch", "Substandard", "Doubtful", "Loss"]
STATUS_OPTIONS = ["Active", "Pending Branch Approval", "Pending Credit Approval", "Rejected", "Closed", "Disbursed"]

def _branches():
    # 70+ pseudo branches across counties
    counties = ["Nairobi","Mombasa","Kisumu","Nakuru","Eldoret","Kakamega","Meru","Nyeri","Thika","Machakos",
                "Naivasha","Kericho","Embu","Kitale","Malindi","Kisii","Garissa","Wajir","Narok","Isiolo",
                "Nanyuki","Voi","Kilifi","Oyugis","Homa Bay","Siaya","Busia","Bungoma","Migori","Keroka",
                "Litein","Bomet","Kapsabet","Lodwar","Marsabit","Maua","Chuka","Mtwapa","Ukunda","Tala",
                "Kajiado","Kimilili","Kanduyi","Kabarnet","Marigat","Muhoroni","Awendo","Bondo","Keroka2","Kajiado2",
                "Kangemi","Kawangware","Gikomba","Kayole","Dandora","Kibera","Ruiru","Juja","Kikuyu","Limuru",
                "Karatina","Othaya","Nyahururu","Gilgil","Eldama Ravine","Sotik","Olkalou","Thika2","Mlolongo","Syokimau",
                "Ruaka","Kitengela","Rongai","Athi River","Thika Road-Mall","Two Rivers"]
    return counties

def _skewed_amount(n, low=5000, high=100000, skew=3.0):
    # more small loans: draw from log-normal-ish
    vals = np.random.lognormal(mean=np.log(20000), sigma=skew, size=n)
    vals = np.clip(vals, low, high)
    return vals.astype(int)

def make_name():
    return f"{random.choice(ENGLISH_FIRST)} {random.choice(KENYAN_SURNAMES)}"

def generate_id_number():
    """Generate realistic Kenyan ID number format"""
    return f"{random.randint(10000000, 39999999)}"

def generate_ref_number():
    """Generate reference number like in the screenshot"""
    return f"25{random.randint(1000000000, 9999999999)}"

def generate(n=2000, female_bias=0.62, small_business_bias=0.6, seed=None,
             fraud_rate=0.02):
    if seed is not None:
        np.random.seed(seed); random.seed(seed)
    branches = _branches()
    rows = []
    base_date = datetime.today() - timedelta(days=365)
    amounts = _skewed_amount(n)
    for i in range(n):
        gender = np.random.choice(["F","M"], p=[female_bias, 1-female_bias])
        occ = np.random.choice(OCCUPATIONS, p=[0.18,0.17,0.15,0.12,0.1,0.08,0.07,0.05,0.04,0.04])
        dependents = np.random.choice([0,1,2,3,4,5], p=[0.1,0.2,0.28,0.24,0.12,0.06])
        income = int(np.random.gamma(4, 8000) + (0 if gender=="F" else 2000))
        loan_amount = int(amounts[i])
        branch = random.choice(branches)
        product = random.choice(PRODUCTS)
        age = int(np.clip(np.random.normal(34, 8), 18, 65))
        # repayment label (1 good / 0 default)
        base_prob = 0.72 \
            + 0.000002*(max(0, 150000 - loan_amount)) \
            + 0.05*(gender=="F") \
            + 0.04*(occ in ["mama mboga","shop owner","teacher","farmer"]) \
            - 0.06*(dependents>=4)
        base_prob = max(0.05, min(0.95, base_prob))
        repay_good = np.random.rand() < base_prob
        # Updated status to match real loan book
        status = np.random.choice(STATUS_OPTIONS, p=[0.35, 0.15, 0.10, 0.12, 0.18, 0.10])
        
        # Loan type based on product and other factors
        if "GROUP" in product:
            loan_type = "Group"
        elif "EMERGENCY" in product:
            loan_type = "Emergency"
        elif income > 80000:
            loan_type = np.random.choice(["Normal", "Salary Advance"], p=[0.7, 0.3])
        else:
            loan_type = "Normal"
        
        # Loan health based on repayment behavior and age of loan
        if repay_good:
            loan_health = np.random.choice(["Performing", "Watch"], p=[0.85, 0.15])
        else:
            loan_health = np.random.choice(["Watch", "Substandard", "Doubtful", "Loss"], p=[0.3, 0.35, 0.25, 0.1])

        # fraud injection
        is_fraud = np.random.rand() < fraud_rate
        if is_fraud:
            income = max(1000, income - np.random.randint(10000,30000))
            loan_amount = min(100000, loan_amount + np.random.randint(15000,40000))
            repay_good = False

        rows.append({
            "customer_name": make_name(),
            "id_reg_number": generate_id_number(),
            "gender": gender,
            "age": age,
            "dependents": dependents,
            "occupation": occ,
            "income": income,
            "branch": branch,
            "product": product,
            "loan_amount": loan_amount,
            "ref_number": generate_ref_number(),
            "loan_type": loan_type,
            "status": status,
            "loan_health": loan_health,
            "repay_good": int(repay_good),
            "is_fraud": int(is_fraud),
            "created_date": (base_date + timedelta(days=np.random.randint(0,365))).date().isoformat(),
            "application_date": (base_date + timedelta(days=np.random.randint(0,365))).date().isoformat()
        })
    df = pd.DataFrame(rows)
    return df