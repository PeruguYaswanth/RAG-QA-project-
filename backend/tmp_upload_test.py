import urllib.request
from urllib.request import Request

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
headers = {'Content-Type': f'multipart/form-data; boundary={boundary}'}

pdf = b'%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids [3 0 R]>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox [0 0 200 200]/Contents 4 0 R>>endobj\n4 0 obj<</Length 55>>stream\nBT /F1 24 Tf 72 120 Td (Hello) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000111 00000 n \n0000000190 00000 n \ntrailer<</Root 1 0 R/Size 5>>\nstartxref\n272\n%%EOF\n'

parts = []
for i in range(2):
    parts.append(b'--' + boundary.encode())
    parts.append(f'Content-Disposition: form-data; name="files"; filename="test{i}.pdf"'.encode())
    parts.append(b'Content-Type: application/pdf\r\n')
    parts.append(pdf)
parts.append(b'--' + boundary.encode() + b'--')

body = b'\r\n'.join(parts)
req = Request('http://127.0.0.1:8000/api/upload', data=body, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(resp.status)
        print(resp.read().decode())
except Exception as e:
    print('ERROR:', e)
