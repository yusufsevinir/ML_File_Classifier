import random
from datetime import datetime, timedelta
import faker
import os
import json

fake = faker.Faker()

def load_templates(doc_type):
    """Load template files from the root/files directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(
        os.path.dirname(current_dir),  
        'training',
        'templates',
        f'{doc_type}_templates.json'
    )
    try:
        with open(template_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise Exception(f"Template file not found: {template_path}")

def generate_bank_statement():
    """Generate synthetic bank statement data"""
    templates = load_templates('bank_statement')
    transactions = []
    balance = random.uniform(1000, 10000)
    date = datetime.now() - timedelta(days=30)
    
    for _ in range(random.randint(10, 20)):
        amount = random.uniform(-500, 1000)
        transaction = {
            'date': date.strftime('%Y-%m-%d'),
            'description': random.choice(templates['transaction_descriptions']),
            'amount': round(amount, 2),
            'balance': round(balance + amount, 2)
        }
        balance += amount
        date += timedelta(days=random.randint(1, 3))
        transactions.append(transaction)
    
    return {
        'account_number': f"****{random.randint(1000, 9999)}",
        'statement_period': f"{(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
        'transactions': transactions
    }

def generate_invoice():
    """Generate synthetic invoice data"""
    templates = load_templates('invoice')
    items = []
    total = 0
    
    for _ in range(random.randint(1, 5)):
        quantity = random.randint(1, 10)
        unit_price = random.uniform(10, 500)
        amount = quantity * unit_price
        total += amount
        
        items.append({
            'description': random.choice(templates['item_descriptions']),
            'quantity': quantity,
            'unit_price': round(unit_price, 2),
            'amount': round(amount, 2)
        })
    
    return {
        'invoice_number': f"INV-{random.randint(1000, 9999)}",
        'date': datetime.now().strftime('%Y-%m-%d'),
        'due_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
        'company_name': fake.company(),
        'items': items,
        'total': round(total, 2)
    }

def generate_drivers_license():
    """Generate synthetic driver's license data"""
    templates = load_templates('drivers_license')
    return {
        'license_number': f"DL{random.randint(100000, 999999)}",
        'name': fake.name(),
        'address': fake.address(),
        'date_of_birth': fake.date_of_birth(minimum_age=18, maximum_age=80).strftime('%Y-%m-%d'),
        'issue_date': (datetime.now() - timedelta(days=random.randint(0, 1825))).strftime('%Y-%m-%d'),
        'expiry_date': (datetime.now() + timedelta(days=random.randint(365, 1825))).strftime('%Y-%m-%d'),
        'class': random.choice(templates['license_classes']),
        'restrictions': random.choice(templates['restrictions'])
    }

def create_synthetic_dataset(num_samples=100):
    """Create a synthetic dataset with mixed document types"""
    dataset = []
    
    for _ in range(num_samples):
        doc_type = random.choice(['bank_statement', 'invoice', 'drivers_license'])
        
        if doc_type == 'bank_statement':
            data = generate_bank_statement()
            text = f"Bank Statement\nAccount: {data['account_number']}\nPeriod: {data['statement_period']}\n"
            for trans in data['transactions']:
                text += f"{trans['date']} - {trans['description']}: ${trans['amount']}\n"
        
        elif doc_type == 'invoice':
            data = generate_invoice()
            text = f"Invoice #{data['invoice_number']}\nDate: {data['date']}\nCompany: {data['company_name']}\n"
            for item in data['items']:
                text += f"{item['description']} - Qty: {item['quantity']} x ${item['unit_price']} = ${item['amount']}\n"
            text += f"Total: ${data['total']}"
        
        else:  # drivers_license
            data = generate_drivers_license()
            text = f"Driver's License\nNumber: {data['license_number']}\nName: {data['name']}\n"
            text += f"DOB: {data['date_of_birth']}\nClass: {data['class']}\nRestrictions: {data['restrictions']}"
        
        dataset.append({
            'text': text,
            'label': doc_type,
            'metadata': data
        })
    
    return dataset

if __name__ == '__main__':
    # Generate sample dataset
    samples = create_synthetic_dataset(5)
    
    # Print example
    for sample in samples:
        print(f"\nDocument Type: {sample['label']}")
        print("Text Content:")
        print(sample['text'])
        print("-" * 50)
