import requests
from pathlib import Path

url = 'http://127.0.0.1:8000/api/upload'

files = {
    'files': (
        'test0.pdf',
        b'%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids [3 0 R]>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox [0 0 200 200]/Contents 4 0 R>>endobj\n4 0 obj<</Length 55>>stream\nBT /F1 24 Tf 72 120 Td (Hello) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000111 00000 n \n0000000190 00000 n \ntrailer<</Root 1 0 R/Size 5>>\nstartxref\n272\n%%EOF\n',
        'application/pdf'
    )
}
files2 = [
    ('files', ('test0.pdf', files['files'][1], 'application/pdf')),
    ('files', ('test1.pdf', files['files'][1], 'application/pdf')),
]

resp = requests.post(url, files=files2, timeout=30)
print(resp.status_code)
print(resp.headers)
print(resp.text)
